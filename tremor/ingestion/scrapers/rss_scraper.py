"""RSS news feed scraper for geopolitical events.

Polls Reuters and AP RSS feeds for market-relevant news, fetches the
full article text with Playwright, and extracts structured fields via
the LLM extractor.

Sources:
- Reuters top news RSS
- AP top news RSS (via rsshub or direct)
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from tremor.ingestion.base import BaseIngester, EventPayload
from tremor.ingestion.scrapers.browser import BrowserManager
from tremor.ingestion.scrapers.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)

RSS_FEEDS = {
    "Reuters": "https://feeds.reuters.com/reuters/topNews",
    "AP": "https://rsshub.app/apnews/topics/ap-top-news",
}

# Keywords that suggest market-moving geopolitical relevance
RELEVANCE_KEYWORDS = {
    "fed", "federal reserve", "interest rate", "inflation", "gdp",
    "tariff", "sanction", "trade war", "war", "conflict", "election",
    "opec", "oil", "recession", "default", "debt ceiling", "treasury",
    "central bank", "ecb", "bank of england", "bank of japan",
    "executive order", "president", "congress", "senate",
}

# LLM schema for geopolitical news articles
GEO_SCHEMA = {
    "summary_text": "2-3 sentence factual summary of the event and its potential market impact",
    "event_category": "conflict | sanctions | election | policy | trade | natural_disaster | financial | other",
    "countries_involved": "array of country names as strings",
    "severity": "low | medium | high",
    "market_relevance": "fx | rates | equities | commodities | broad | none",
    "affected_sectors": "array of sector names (e.g. ['energy', 'technology'])",
    "vix_before": null,
    "vix_after": null,
    "spread_before": null,
    "spread_after": null,
}

# Remove null literal — just use the schema as a plain dict with string descriptions
GEO_SCHEMA = {
    "summary_text": "2-3 sentence factual summary of the event and its potential market impact",
    "event_category": "one of: conflict | sanctions | election | policy | trade | natural_disaster | financial | other",
    "countries_involved": "array of country names",
    "severity": "one of: low | medium | high",
    "market_relevance": "one of: fx | rates | equities | commodities | broad | none",
    "affected_sectors": "array of sector names",
}


class RssScraper(BaseIngester):
    """Poll RSS feeds for geopolitical news and extract structured fields."""

    def __init__(self, api_key: Optional[str] = None):
        self._extractor = LLMExtractor(api_key=api_key)

    async def fetch(
        self,
        feed_name: str = "Reuters",
        limit: int = 10,
        relevance_filter: bool = True,
    ) -> list[EventPayload]:
        """Fetch and process articles from an RSS feed.

        Args:
            feed_name: Key in RSS_FEEDS dict ("Reuters" or "AP")
            limit: Max number of articles to process
            relevance_filter: If True, skip articles with no market-relevant keywords

        Returns:
            List of EventPayload with summary_text and category fields populated.
            vix_before/after and spread_before/after are NOT populated here —
            the market data fetcher enriches these in a post-processing step.
        """
        feed_url = RSS_FEEDS.get(feed_name)
        if not feed_url:
            raise ValueError(f"Unknown feed: {feed_name}. Options: {list(RSS_FEEDS)}")

        items = await self._fetch_rss(feed_url)
        if not items:
            return []

        if relevance_filter:
            items = [i for i in items if self._is_relevant(i.get("title", "") + " " + i.get("description", ""))]

        items = items[:limit]
        logger.info(f"{feed_name}: processing {len(items)} relevant articles")

        payloads = []
        async with BrowserManager() as browser:
            for item in items:
                payload = await self._process_item(item, feed_name, browser)
                if payload:
                    payloads.append(payload)

        return payloads

    async def _fetch_rss(self, feed_url: str) -> list[dict]:
        """Fetch and parse an RSS feed. Returns list of item dicts."""
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(feed_url)
            response.raise_for_status()

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            logger.error(f"RSS parse error: {e}")
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = []

        # Standard RSS 2.0
        for item in root.findall(".//item"):
            items.append({
                "title": item.findtext("title", ""),
                "link": item.findtext("link", ""),
                "description": item.findtext("description", ""),
                "pub_date": item.findtext("pubDate", ""),
            })

        # Atom feed fallback
        if not items:
            for entry in root.findall(".//atom:entry", ns):
                link_el = entry.find("atom:link", ns)
                items.append({
                    "title": entry.findtext("atom:title", "", ns),
                    "link": link_el.get("href", "") if link_el is not None else "",
                    "description": entry.findtext("atom:summary", "", ns),
                    "pub_date": entry.findtext("atom:updated", "", ns),
                })

        return items

    async def _process_item(
        self,
        item: dict,
        feed_name: str,
        browser: BrowserManager,
    ) -> Optional[EventPayload]:
        """Scrape article and extract fields for a single RSS item."""
        url = item.get("link", "")
        if not url:
            return None

        html = await browser.fetch_html(url)
        if not html:
            logger.warning(f"Could not fetch article: {url}")
            return None

        fields = self._extractor.extract(html, GEO_SCHEMA, url=url)
        timestamp = self._parse_date(item.get("pub_date", ""))

        summary = fields.get("summary_text") or item.get("title") or "Geopolitical event"
        category = fields.get("event_category", "other")
        severity = fields.get("severity", "medium")

        tags = ["geopolitical", feed_name.lower(), category]
        if severity == "high":
            tags.append("high_severity")

        countries = fields.get("countries_involved") or []
        if isinstance(countries, list):
            tags.extend([c.lower().replace(" ", "_") for c in countries[:3]])

        extra = {
            k: v for k, v in fields.items()
            if k != "summary_text" and v is not None
        }

        return EventPayload(
            event_type="geopolitical",
            event_subtype="news_event",
            timestamp=timestamp,
            description=item.get("title") or summary,
            source_name=feed_name,
            source_url=url,
            summary_text=summary,
            tags=tags,
            extra=extra,
        )

    def _is_relevant(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in RELEVANCE_KEYWORDS)

    def _parse_date(self, date_str: str) -> datetime:
        if not date_str:
            return datetime.now(timezone.utc)
        try:
            return parsedate_to_datetime(date_str).astimezone(timezone.utc)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return datetime.now(timezone.utc)
