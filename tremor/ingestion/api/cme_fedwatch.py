"""CME FedWatch ingester.

Fetches the market-implied expected Fed funds rate from CME FedWatch
tool page. This is used to populate `expected_rate` for fed_announcement
events, which is the other half of the rate surprise calculation.

No API key required — parses the public CME FedWatch page.
The page embeds probability data in JavaScript. This scraper extracts
the probability distribution over rate outcomes and computes the
probability-weighted implied rate.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from tremor.ingestion.base import BaseIngester, EventPayload

logger = logging.getLogger(__name__)

FEDWATCH_URL = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"

# Current lower bound of each rate outcome bucket (in percent)
# as shown on FedWatch. Update if the Fed moves outside this range.
RATE_BUCKETS = {
    "525-550": 5.375,
    "500-525": 5.125,
    "475-500": 4.875,
    "450-475": 4.625,
    "425-450": 4.375,
    "400-425": 4.125,
    "375-400": 3.875,
    "350-375": 3.625,
    "325-350": 3.375,
    "300-325": 3.125,
}


class CmeFedWatchIngester(BaseIngester):
    """Scrape CME FedWatch to compute probability-weighted expected Fed rate."""

    async def fetch(
        self,
        meeting_date: Optional[str] = None,
    ) -> list[EventPayload]:
        """Fetch the current probability-weighted implied rate from FedWatch.

        Args:
            meeting_date: ISO date of the FOMC meeting "YYYY-MM-DD".
                          If None, uses the next upcoming meeting shown on the page.

        Returns:
            A single EventPayload with expected_rate populated.
            This is intended to be merged with a fed_announcement EventPayload
            by the normaliser — not stored as a standalone event.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

        async with httpx.AsyncClient(timeout=30, headers=headers) as client:
            response = await client.get(FEDWATCH_URL)
            response.raise_for_status()
            html = response.text

        expected_rate = self._parse_implied_rate(html)
        if expected_rate is None:
            logger.warning("Could not parse implied rate from FedWatch page")
            return []

        ts = datetime.now(timezone.utc)
        payload = EventPayload(
            event_type="fed_announcement",
            event_subtype="rate_decision",
            timestamp=ts,
            description=f"CME FedWatch implied rate: {expected_rate:.3f}%",
            source_name="CME FedWatch",
            source_url=FEDWATCH_URL,
            expected_rate=expected_rate,
            tags=["fomc", "futures", "expected_rate"],
        )

        logger.info(f"CME FedWatch: implied rate = {expected_rate:.3f}%")
        return [payload]

    def _parse_implied_rate(self, html: str) -> Optional[float]:
        """Extract probability-weighted implied rate from page HTML.

        CME embeds probabilities as JSON in a <script> tag or as
        data attributes on table cells. This method tries several
        extraction strategies.
        """
        # Strategy 1: look for JSON blob with probability data
        json_match = re.search(
            r'"probabilities"\s*:\s*(\{[^}]+\})', html
        )
        if json_match:
            try:
                import json
                probs = json.loads(json_match.group(1))
                return self._weighted_rate(probs)
            except Exception:
                pass

        # Strategy 2: parse the probability table
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_=re.compile(r"fedwatch|probability", re.I))
        if table:
            probs = self._parse_probability_table(table)
            if probs:
                return self._weighted_rate(probs)

        # Strategy 3: look for the "current" implied rate displayed directly
        rate_match = re.search(r"implied\s+rate[:\s]+(\d+\.\d+)%", html, re.I)
        if rate_match:
            return float(rate_match.group(1))

        return None

    def _parse_probability_table(self, table) -> dict[str, float]:
        """Extract outcome → probability from an HTML table."""
        probs = {}
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                value = cells[-1].get_text(strip=True).replace("%", "")
                for bucket_key in RATE_BUCKETS:
                    if bucket_key.replace("-", " - ") in label or bucket_key in label:
                        try:
                            probs[bucket_key] = float(value) / 100
                        except ValueError:
                            pass
        return probs

    def _weighted_rate(self, probs: dict[str, float]) -> Optional[float]:
        """Compute probability-weighted expected rate from outcome distribution."""
        total_weight = 0.0
        weighted_sum = 0.0
        for bucket_key, prob in probs.items():
            midpoint = RATE_BUCKETS.get(bucket_key)
            if midpoint is not None and prob > 0:
                weighted_sum += midpoint * prob
                total_weight += prob
        if total_weight == 0:
            return None
        return weighted_sum / total_weight
