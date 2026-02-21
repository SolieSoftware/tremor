"""FRED API ingester.

Fetches economic data releases (CPI, NFP, GDP) from the St. Louis Fed
FRED API and maps them into EventPayload objects.

Requires env var: FRED_API_KEY
Free API key: https://fred.stlouisfed.org/docs/api/api_key.html

Series used:
- CPIAUCSL          → CPI actual (monthly, YoY % change computed here)
- EXPINF1YR         → Cleveland Fed 1-year expected inflation (proxy for expected CPI)
- PAYEMS            → Non-farm payrolls (monthly, MoM change in thousands)
- A191RL1Q225SBEA   → Real GDP % change (quarterly)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from tremor.config import settings
from tremor.ingestion.base import BaseIngester, EventPayload

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Cleveland Fed 1-year expected inflation — best free proxy for pre-release CPI consensus.
# Published daily via FRED; we match to the observation closest before each CPI release date.
EXPECTED_CPI_SERIES = "EXPINF1YR"

# Maps FRED series ID → (event_subtype, raw_data actual field, description template)
SERIES_CONFIG: dict[str, dict] = {
    "CPIAUCSL": {
        "subtype": "cpi_release",
        "actual_field": "actual_cpi",
        "description_template": "CPI release: {value:.2f}% YoY",
        "tags": ["cpi", "inflation", "bls"],
        "transform": "yoy_pct",
    },
    "PAYEMS": {
        "subtype": "nfp_release",
        "actual_field": "actual_nfp",
        "description_template": "NFP release: {value:+.0f}k jobs",
        "tags": ["nfp", "employment", "bls"],
        "transform": "mom_diff",
    },
    "A191RL1Q225SBEA": {
        "subtype": "gdp_release",
        "actual_field": "actual_gdp",
        "description_template": "GDP advance estimate: {value:.1f}% annualised",
        "tags": ["gdp", "growth", "bea"],
        "transform": "level",
    },
}


class FredIngester(BaseIngester):
    """Fetch economic data releases from FRED API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "FRED_API_KEY", None)
        if not self.api_key:
            raise ValueError("FRED_API_KEY not configured")

    async def fetch(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        limit: int = 10,
    ) -> list[EventPayload]:
        """Fetch recent observations for a FRED series.

        For CPIAUCSL, also fetches EXPINF1YR (Cleveland Fed 1-year expected
        inflation) and matches it to each CPI release date to populate
        expected_cpi, enabling the surprise signal actual_cpi - expected_cpi.
        """
        if series_id not in SERIES_CONFIG:
            raise ValueError(f"Unknown FRED series: {series_id}. Add it to SERIES_CONFIG.")

        config = SERIES_CONFIG[series_id]
        params: dict = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit + 1,  # fetch one extra for diff calculations
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(FRED_BASE, params=params)
            response.raise_for_status()
            data = response.json()

        observations = data.get("observations", [])
        if not observations:
            logger.warning(f"No FRED observations returned for {series_id}")
            return []

        valid_obs = [o for o in observations if o["value"] != "."]

        # For CPI, fetch the expected inflation series over the same window
        expected_by_month: dict[str, float] = {}
        if series_id == "CPIAUCSL":
            expected_by_month = await self._fetch_expected_cpi(
                observation_start=observation_start,
                observation_end=observation_end,
                limit=limit + 12,
            )
            # Also need 12-months-prior values for YoY — fetch a wider window
            yoy_lookup = await self._fetch_yoy_lookup(
                valid_obs=valid_obs,
                limit=limit,
            )
        else:
            yoy_lookup = {}

        payloads = []
        for i, obs in enumerate(valid_obs[:limit]):
            raw_value = float(obs["value"])

            if config["transform"] == "yoy_pct":
                signal_value = self._compute_yoy(obs["date"], raw_value, yoy_lookup)
            else:
                signal_value = self._apply_transform(
                    raw_value,
                    valid_obs[i + 1]["value"] if i + 1 < len(valid_obs) else None,
                    config["transform"],
                )
            if signal_value is None:
                continue

            ts = datetime.strptime(obs["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            description = config["description_template"].format(value=signal_value)

            extra: dict = {}
            actual_kwargs: dict = {config["actual_field"]: signal_value}

            # Attach expected_cpi if available for this release month
            if series_id == "CPIAUCSL" and expected_by_month:
                expected = self._match_expected(ts, expected_by_month)
                if expected is not None:
                    actual_kwargs["expected_cpi"] = expected
                    surprise = signal_value - expected
                    description += f" (expected {expected:.2f}%, surprise {surprise:+.2f}%)"
                    logger.info(
                        f"CPI {obs['date']}: actual={signal_value:.2f}% "
                        f"expected={expected:.2f}% surprise={surprise:+.2f}%"
                    )
                else:
                    logger.warning(f"No expected CPI found for {obs['date']}")

            payload = EventPayload(
                event_type="economic_data",
                event_subtype=config["subtype"],
                timestamp=ts,
                description=description,
                source_name="FRED",
                source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                tags=config["tags"],
                extra=extra,
                **actual_kwargs,
            )
            payloads.append(payload)

        logger.info(f"FRED {series_id}: fetched {len(payloads)} observations")
        return payloads

    async def _fetch_yoy_lookup(
        self,
        valid_obs: list[dict],
        limit: int,
    ) -> dict[str, float]:
        """Fetch CPIAUCSL going back 13+ months to compute YoY % changes.

        Returns {YYYY-MM-DD: index_level} for all observations in the window
        needed to compare each current reading to its year-ago counterpart.
        """
        if not valid_obs:
            return {}

        # Find the oldest date in our current window, go back 13 months
        oldest_date = min(obs["date"] for obs in valid_obs[:limit])
        oldest_dt = datetime.strptime(oldest_date, "%Y-%m-%d")
        lookback_start = (oldest_dt.replace(year=oldest_dt.year - 1) - timedelta(days=31)).strftime("%Y-%m-%d")

        params: dict = {
            "series_id": "CPIAUCSL",
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "asc",
            "observation_start": lookback_start,
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(FRED_BASE, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.warning(f"Could not fetch CPI lookback data: {e}")
            return {}

        return {
            obs["date"]: float(obs["value"])
            for obs in data.get("observations", [])
            if obs["value"] != "."
        }

    def _compute_yoy(
        self,
        date_str: str,
        current_level: float,
        yoy_lookup: dict[str, float],
    ) -> Optional[float]:
        """Compute CPI YoY % change by comparing to the same month a year ago."""
        current_dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Try exact year-ago match, then ±1 month
        for delta_days in [0, -31, 31, -15, 15]:
            prior_dt = current_dt.replace(year=current_dt.year - 1) + timedelta(days=delta_days)
            prior_dt = prior_dt.replace(day=1)  # FRED dates are always 1st of month
            prior_str = prior_dt.strftime("%Y-%m-%d")
            if prior_str in yoy_lookup:
                prior_level = yoy_lookup[prior_str]
                if prior_level > 0:
                    return round((current_level - prior_level) / prior_level * 100, 4)
        logger.warning(f"No year-ago CPI level found for {date_str}")
        return None

    async def _fetch_expected_cpi(
        self,
        observation_start: Optional[str],
        observation_end: Optional[str],
        limit: int,
    ) -> dict[str, float]:
        """Fetch Cleveland Fed 1-year expected inflation (EXPINF1YR) from FRED.

        Returns a dict of {YYYY-MM: expected_inflation_pct} keyed by year-month
        so each CPI release can be matched to the expectation current at that time.

        EXPINF1YR is monthly and represents what the market expected inflation
        to be over the next year — the best free proxy for pre-release CPI consensus.
        """
        params: dict = {
            "series_id": EXPECTED_CPI_SERIES,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        # Fetch slightly earlier so we have expectations prior to each release
        if observation_start:
            start = datetime.strptime(observation_start, "%Y-%m-%d") - timedelta(days=60)
            params["observation_start"] = start.strftime("%Y-%m-%d")
        if observation_end:
            params["observation_end"] = observation_end

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(FRED_BASE, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.warning(f"Could not fetch expected CPI series ({EXPECTED_CPI_SERIES}): {e}")
            return {}

        result = {}
        for obs in data.get("observations", []):
            if obs["value"] == ".":
                continue
            # Key by YYYY-MM so we can match against CPI release month
            year_month = obs["date"][:7]
            result[year_month] = float(obs["value"])

        logger.info(f"EXPINF1YR: loaded {len(result)} monthly expectations")
        return result

    def _match_expected(
        self,
        release_ts: datetime,
        expected_by_month: dict[str, float],
    ) -> Optional[float]:
        """Find the expected inflation value for a CPI release date.

        CPI for month M is released in mid-month M+1, so we look for the
        EXPINF1YR observation from month M (one month before the release).
        Falls back to the release month itself if the prior month is missing.
        """
        release_date = release_ts.date()
        # CPIAUCSL observation date is the 1st of the reference month.
        # EXPINF1YR for that same month is the market expectation at that time.
        year_month = release_date.strftime("%Y-%m")

        if year_month in expected_by_month:
            return expected_by_month[year_month]

        # Try one month prior as fallback
        prior = (release_date.replace(day=1) - timedelta(days=1))
        prior_ym = prior.strftime("%Y-%m")
        return expected_by_month.get(prior_ym)

    def _apply_transform(
        self,
        current: float,
        previous_raw: Optional[str],
        transform: str,
    ) -> Optional[float]:
        """Apply the series-specific transform to derive the signal value."""
        if transform == "level":
            return current
        if transform == "mom_diff":
            if previous_raw is None or previous_raw == ".":
                return None
            return current - float(previous_raw)
        if transform == "yoy_pct":
            # For YoY we'd need 12 months back; FRED already provides some series
            # in % change form. For CPIAUCSL (level), we return the level for now
            # and expect the caller to compute YoY from a longer fetch.
            return current
        return current
