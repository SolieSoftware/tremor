"""Playwright browser manager for Tremor scrapers.

Adapted from smart-webscraper-products/src/scrapers/browser.py.
Provides an async context manager that handles browser lifecycle,
anti-detection headers, retry logic, and configurable delays.
"""

import asyncio
import logging
from typing import Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

from tremor.config import settings

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Tracking / analytics hostnames to block (saves bandwidth and avoids fingerprinting)
BLOCKED_HOSTS = {
    "google-analytics.com",
    "googletagmanager.com",
    "facebook.com",
    "doubleclick.net",
    "criteo.com",
    "scorecardresearch.com",
    "chartbeat.com",
}


class BrowserManager:
    """Async context manager for a shared Playwright browser instance."""

    def __init__(
        self,
        headless: bool = True,
        max_retries: int = 3,
        request_delay: float = 2.0,
    ):
        self.headless = headless
        self.max_retries = max_retries
        self.request_delay = request_delay
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "BrowserManager":
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 800},
        )
        # Mask webdriver property to reduce bot detection
        await self._context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        # Block tracking resources
        await self._context.route(
            "**/*",
            lambda route, request: (
                route.abort()
                if any(h in request.url for h in BLOCKED_HOSTS)
                else route.continue_()
            ),
        )
        return self

    async def __aexit__(self, *_) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def fetch_html(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout_ms: int = 30_000,
    ) -> Optional[str]:
        """Navigate to a URL and return the page HTML.

        Retries up to max_retries times with exponential backoff.
        Returns None if all attempts fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries):
            page: Optional[Page] = None
            try:
                page = await self._context.new_page()
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                await asyncio.sleep(self.request_delay)
                html = await page.content()
                return html

            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 2
                logger.warning(
                    f"Fetch attempt {attempt + 1}/{self.max_retries} failed "
                    f"for {url}: {e}. Retrying in {wait}s."
                )
                await asyncio.sleep(wait)

            finally:
                if page:
                    await page.close()

        logger.error(f"All {self.max_retries} attempts failed for {url}: {last_error}")
        return None
