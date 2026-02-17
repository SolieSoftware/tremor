"""Polygon.io earnings ingester.

Fetches actual and expected EPS from Polygon.io for public companies and
maps them into EventPayload objects with event_type="earnings".

Requires env var: POLYGON_API_KEY
Free tier: https://polygon.io/dashboard (delayed data, limited endpoints)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from tremor.config import settings
from tremor.ingestion.base import BaseIngester, EventPayload

logger = logging.getLogger(__name__)

POLYGON_BASE = "https://api.polygon.io"


class PolygonEarningsIngester(BaseIngester):
    """Fetch earnings actual vs. expected EPS from Polygon.io."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "POLYGON_API_KEY", None)
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not configured")

    async def fetch(
        self,
        ticker: str,
        timeframe: str = "quarterly",
        limit: int = 8,
    ) -> list[EventPayload]:
        """Fetch earnings results for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g. "AAPL")
            timeframe: "quarterly" or "annual"
            limit: Number of earnings periods to return

        Returns:
            List of EventPayload with actual_eps and expected_eps populated.
        """
        url = f"{POLYGON_BASE}/vX/reference/financials"
        params = {
            "ticker": ticker,
            "timeframe": timeframe,
            "limit": limit,
            "apiKey": self.api_key,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not results:
            logger.warning(f"No Polygon earnings data for {ticker}")
            return []

        # Fetch analyst estimates separately
        estimates = await self._fetch_estimates(ticker, limit)

        payloads = []
        for result in results:
            period_end = result.get("end_date")
            if not period_end:
                continue

            ts = datetime.strptime(period_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            # EPS from income statement
            income = result.get("financials", {}).get("income_statement", {})
            eps_data = income.get("basic_earnings_per_share", {})
            actual_eps = eps_data.get("value")

            # Match to estimate by period
            expected_eps = estimates.get(period_end)

            description = (
                f"{ticker} Q earnings: actual EPS {actual_eps}, "
                f"expected {expected_eps}"
            )

            payload = EventPayload(
                event_type="earnings",
                event_subtype="eps_release",
                timestamp=ts,
                description=description,
                source_name="Polygon.io",
                source_url=f"https://polygon.io/stocks/{ticker}",
                actual_eps=float(actual_eps) if actual_eps is not None else None,
                expected_eps=float(expected_eps) if expected_eps is not None else None,
                tags=[ticker.lower(), "earnings", "eps"],
                extra={
                    "ticker": ticker,
                    "fiscal_period": result.get("fiscal_period"),
                    "fiscal_year": result.get("fiscal_year"),
                },
            )
            payloads.append(payload)

        logger.info(f"Polygon {ticker}: fetched {len(payloads)} earnings periods")
        return payloads

    async def _fetch_estimates(self, ticker: str, limit: int) -> dict[str, float]:
        """Fetch analyst EPS estimates, keyed by period end date."""
        url = f"{POLYGON_BASE}/v2/reference/financials/{ticker}"
        params = {
            "limit": limit,
            "type": "Q",
            "apiKey": self.api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                if response.status_code == 404:
                    return {}
                response.raise_for_status()
                data = response.json()

            estimates = {}
            for item in data.get("results", []):
                period = item.get("period")
                eps_est = item.get("EPSReportedConsensus") or item.get("EPSEstimated")
                if period and eps_est is not None:
                    estimates[period] = float(eps_est)
            return estimates

        except Exception as e:
            logger.warning(f"Could not fetch estimates for {ticker}: {e}")
            return {}
