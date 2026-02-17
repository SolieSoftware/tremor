"""White House briefings and statements scraper.

Scrapes https://www.whitehouse.gov/briefings-statements/ for presidential
announcements, executive orders, and policy statements.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

from bs4 import BeautifulSoup

from tremor.ingestion.base import BaseIngester, EventPayload
from tremor.ingestion.scrapers.browser import BrowserManager
from tremor.ingestion.scrapers.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

WH_BRIEFINGS_URL = "https://www.whitehouse.gov/briefings-statements/"

WH_SCHEMA = {
    "summary_text": "2-3 sentence factual summary of the announcement and its policy implications",
    "policy_area": "one of: trade | defense | fiscal | monetary | regulatory | immigration | energy | technology | other",
    "executive_action": "true if this is an executive order or presidential action, else false",
    "affected_sectors": "array of economic sector names (e.g. ['energy', 'agriculture'])",
    "countries_involved": "array of country names if relevant",
    "event_category": "one of: executive_order | speech | statement | press_briefing | other",
    "market_relevance": "one of: fx | rates | equities | commodities | broad | none",
    "severity": "one of: low | medium | high — based on likely market impact",
}


class WhiteHouseScraper(BaseIngester):
    """Scrape White House briefings and statements."""

    def __init__(self, api_key: Optional[str] = None):
        self._extractor = LLMExtractor(api_key=api_key)

    async def fetch(
        self,
        limit: int = 5,
    ) -> list[EventPayload]:
        """Fetch the most recent White House briefings and statements.

        Args:
            limit: Number of recent statements to process

        Returns:
            List of EventPayload with geopolitical event type.
        """
        async with BrowserManager() as browser:
            index_html = await browser.fetch_html(WH_BRIEFINGS_URL)
            if not index_html:
                logger.error("Failed to fetch White House briefings index")
                return []

            article_urls = self._extract_article_urls(index_html, limit)
            logger.info(f"White House: found {len(article_urls)} articles")

            payloads = []
            for url, title, date_str in article_urls:
                html = await browser.fetch_html(url)
                if not html:
                    continue

                fields = self._extractor.extract(html, WH_SCHEMA, url=url)
                timestamp = self._parse_date(date_str)
                payload = self._build_payload(url, title, timestamp, fields)
                if payload:
                    payloads.append(payload)

        return payloads

    def _extract_article_urls(
        self, html: str, limit: int
    ) -> list[tuple[str, str, str]]:
        """Parse the briefings index page for article URLs, titles, and dates."""
        soup = BeautifulSoup(html, "lxml")
        results = []
        seen = set()

        for article in soup.find_all("article"):
            if len(results) >= limit:
                break

            link = article.find("a", href=re.compile(r"/briefings-statements/"))
            if not link:
                continue

            href = link.get("href", "")
            if href in seen:
                continue
            seen.add(href)

            title = link.get_text(strip=True)

            date_el = article.find("time")
            date_str = date_el.get("datetime", "") if date_el else ""

            full_url = href if href.startswith("http") else "https://www.whitehouse.gov" + href
            results.append((full_url, title, date_str))

        return results

    def _build_payload(
        self,
        url: str,
        title: str,
        timestamp: datetime,
        fields: dict,
    ) -> Optional[EventPayload]:
        summary = fields.get("summary_text") or title
        category = fields.get("event_category", "statement")
        policy_area = fields.get("policy_area", "other")
        severity = fields.get("severity", "medium")

        tags = ["geopolitical", "whitehouse", category, policy_area]
        if severity == "high":
            tags.append("high_severity")
        if fields.get("executive_action") is True:
            tags.append("executive_order")

        extra = {
            k: v for k, v in fields.items()
            if k != "summary_text" and v is not None
        }

        return EventPayload(
            event_type="geopolitical",
            event_subtype="presidential_statement",
            timestamp=timestamp,
            description=title or summary,
            source_name="White House",
            source_url=url,
            summary_text=summary,
            tags=tags,
            extra=extra,
        )

    def _parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return datetime.now(timezone.utc)
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)
