"""Federal Reserve press release scraper.

Scrapes FOMC statements and other Fed press releases from federalreserve.gov
and extracts structured fields via the LLM extractor.

Handles:
- Rate decision press releases (most structured)
- FOMC minutes
- Governor speeches
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from tremor.ingestion.base import BaseIngester, EventPayload
from tremor.ingestion.scrapers.browser import BrowserManager
from tremor.ingestion.scrapers.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

FED_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendar.htm"
FED_SPEECHES_URL = "https://www.federalreserve.gov/newsevents/speeches.htm"

# LLM schema for rate decision press releases
RATE_DECISION_SCHEMA = {
    "summary_text": "2-3 sentence summary of the decision and key policy signals",
    "actual_rate": "midpoint of the new target rate range as a float (e.g. 5.375)",
    "rate_action": "hike | cut | hold",
    "vote_tally": "e.g. '12-0' or '11-1'",
    "tone": "hawkish | dovish | neutral",
    "inflation_assessment": "brief string describing Fed view on inflation",
    "employment_assessment": "brief string describing Fed view on employment",
}

# LLM schema for speeches / minutes
SPEECH_SCHEMA = {
    "summary_text": "2-3 sentence factual summary of key policy signals",
    "tone": "hawkish | dovish | neutral",
    "key_topics": "list of topics as array of strings",
    "rate_bias": "hike | hold | cut | unclear",
    "speaker": "name of the speaker if identifiable, else null",
}


class FedScraper(BaseIngester):
    """Scrape Federal Reserve press releases and speeches."""

    def __init__(self, api_key: Optional[str] = None):
        self._extractor = LLMExtractor(api_key=api_key)

    async def fetch(
        self,
        url: str,
        subtype: str = "rate_decision",
    ) -> list[EventPayload]:
        """Scrape a single Fed press release URL.

        Args:
            url: Direct URL to the press release page
            subtype: "rate_decision", "minutes", or "speech"

        Returns:
            List containing a single EventPayload.
        """
        schema = RATE_DECISION_SCHEMA if subtype == "rate_decision" else SPEECH_SCHEMA

        async with BrowserManager() as browser:
            html = await browser.fetch_html(url)

        if not html:
            logger.error(f"Failed to fetch Fed page: {url}")
            return []

        fields = self._extractor.extract(html, schema, url=url)

        timestamp = self._extract_date_from_url(url) or datetime.now(timezone.utc)
        description = fields.get("summary_text") or f"Federal Reserve {subtype}"

        actual_rate: Optional[float] = None
        if fields.get("actual_rate") is not None:
            try:
                actual_rate = float(fields["actual_rate"])
            except (TypeError, ValueError):
                pass

        extra = {k: v for k, v in fields.items() if k not in ("actual_rate", "summary_text") and v is not None}

        payload = EventPayload(
            event_type="fed_announcement",
            event_subtype=subtype,
            timestamp=timestamp,
            description=description,
            source_name="Federal Reserve",
            source_url=url,
            actual_rate=actual_rate,
            summary_text=fields.get("summary_text"),
            tags=["fomc", "fed", subtype],
            extra=extra,
        )

        return [payload]

    async def fetch_recent_releases(self, limit: int = 5) -> list[EventPayload]:
        """Fetch the most recent FOMC press release URLs from the calendar page
        and scrape each one.
        """
        async with BrowserManager() as browser:
            html = await browser.fetch_html(FED_CALENDAR_URL)

        if not html:
            return []

        urls = self._extract_release_urls(html, limit)
        payloads = []
        for url in urls:
            results = await self.fetch(url, subtype="rate_decision")
            payloads.extend(results)
        return payloads

    def _extract_release_urls(self, html: str, limit: int) -> list[str]:
        """Parse the FOMC calendar page to find press release links."""
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        links = soup.find_all("a", href=re.compile(r"/newsevents/pressreleases/monetary\d+"))
        base = "https://www.federalreserve.gov"
        seen = set()
        urls = []
        for link in links:
            href = link.get("href", "")
            full = base + href if href.startswith("/") else href
            if full not in seen:
                seen.add(full)
                urls.append(full)
            if len(urls) >= limit:
                break
        return urls

    def _extract_date_from_url(self, url: str) -> Optional[datetime]:
        """Try to parse a date from URL patterns like /monetary20240131a.htm"""
        match = re.search(r"(\d{8})", url)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return None
