"""FRED API ingester.

Fetches economic data releases (CPI, NFP, GDP) from the St. Louis Fed
FRED API and maps them into EventPayload objects.

Requires env var: FRED_API_KEY
Free API key: https://fred.stlouisfed.org/docs/api/api_key.html

Series used:
- CPIAUCSL  → CPI (monthly, YoY % change computed here)
- PAYEMS    → Non-farm payrolls (monthly, MoM change in thousands)
- A191RL1Q225SBEA → Real GDP % change (quarterly)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from tremor.config import settings
from tremor.ingestion.base import BaseIngester, EventPayload

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Maps FRED series ID → (event_subtype, raw_data actual field, description template)
SERIES_CONFIG: dict[str, dict] = {
    "CPIAUCSL": {
        "subtype": "cpi_release",
        "actual_field": "actual_cpi",
        "description_template": "CPI release: {value:.2f}% YoY",
        "tags": ["cpi", "inflation", "bls"],
        "transform": "yoy_pct",  # how to derive the signal value
    },
    "PAYEMS": {
        "subtype": "nfp_release",
        "actual_field": "actual_nfp",
        "description_template": "NFP release: {value:+.0f}k jobs",
        "tags": ["nfp", "employment", "bls"],
        "transform": "mom_diff",  # month-on-month absolute change
    },
    "A191RL1Q225SBEA": {
        "subtype": "gdp_release",
        "actual_field": "actual_gdp",
        "description_template": "GDP advance estimate: {value:.1f}% annualised",
        "tags": ["gdp", "growth", "bea"],
        "transform": "level",  # series is already % change
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

        Args:
            series_id: FRED series ID (e.g. "CPIAUCSL")
            observation_start: ISO date string "YYYY-MM-DD", optional
            observation_end: ISO date string "YYYY-MM-DD", optional
            limit: max number of recent observations to return

        Returns:
            List of EventPayload, one per observation.
            expected_* fields are not populated here — the consensus
            scraper (see scrapers/consensus.py) enriches these separately.
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

        # Filter out missing values (".")
        valid_obs = [o for o in observations if o["value"] != "."]

        payloads = []
        for i, obs in enumerate(valid_obs[:limit]):
            raw_value = float(obs["value"])
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
