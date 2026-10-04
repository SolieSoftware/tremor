from unittest.mock import MagicMock, patch
from datetime import datetime

import httpx

from tremor.market_data.fetcher import fetch_daily_node_data


def test_fred_node_data_parses_csv_and_drops_missing():
    csv = "observation_date,DGS10\n2025-01-02,4.57\n2025-01-03,.\n2025-01-06,4.62\n"
    response = MagicMock(text=csv)
    with patch("tremor.market_data.fetcher.httpx.get", return_value=response) as get:
        series = fetch_daily_node_data("d_treasury_10y", datetime(2025, 1, 1), datetime(2025, 1, 10))

    assert get.call_args.kwargs["params"]["id"] == "DGS10"
    assert list(series.values) == [4.57, 4.62]


def test_fred_node_data_returns_empty_on_http_error():
    with patch("tremor.market_data.fetcher.httpx.get", side_effect=httpx.ConnectError("down")):
        series = fetch_daily_node_data("d_credit_spread", datetime(2025, 1, 1), datetime(2025, 1, 10))
    assert series.empty


def test_scraper_modules_import():
    import tremor.ingestion.scrapers.fed_scraper  # noqa: F401
    import tremor.ingestion.scrapers.rss_scraper  # noqa: F401
    import tremor.ingestion.scrapers.whitehouse_scraper  # noqa: F401


def test_fed_release_urls_newest_first():
    from tremor.ingestion.scrapers.fed_scraper import FedScraper

    html = """
      <a href="/newsevents/pressreleases/monetary20260128a.htm">Jan</a>
      <a href="/newsevents/pressreleases/monetary20260128a1.htm">Implementation note</a>
      <a href="/newsevents/pressreleases/monetary20260916a.htm">Sep</a>
      <a href="/newsevents/pressreleases/monetary20260429a.htm">Apr</a>
    """
    scraper = FedScraper.__new__(FedScraper)
    urls = scraper._extract_release_urls(html, limit=2)
    assert urls == [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260916a.htm",
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260429a.htm",
    ]


def test_whitehouse_parses_post_blocks():
    from tremor.ingestion.scrapers.whitehouse_scraper import WhiteHouseScraper

    html = """
      <ul>
        <li class="wp-block-post">
          <h2><a href="https://www.whitehouse.gov/briefings-statements/2026/10/some-statement/">Some Statement</a></h2>
          <a href="https://www.whitehouse.gov/briefings-statements/" rel="tag">Briefings &amp; Statements</a>
          <time datetime="2026-10-02T14:28:12-04:00">October 2, 2026</time>
        </li>
      </ul>
    """
    scraper = WhiteHouseScraper.__new__(WhiteHouseScraper)
    assert scraper._extract_article_urls(html, limit=5) == [(
        "https://www.whitehouse.gov/briefings-statements/2026/10/some-statement/",
        "Some Statement",
        "2026-10-02T14:28:12-04:00",
    )]


def test_null_llm_fields_do_not_break_tags():
    """The extractor returns null for unknown fields; tags must stay valid strings."""
    from tremor.ingestion.normaliser import normalise
    from tremor.ingestion.scrapers.whitehouse_scraper import WhiteHouseScraper

    fields = {"summary_text": None, "event_category": None, "policy_area": None, "severity": None}
    scraper = WhiteHouseScraper.__new__(WhiteHouseScraper)
    payload = scraper._build_payload("https://example.gov/x", "Title", datetime(2026, 1, 1), fields)

    event = normalise(payload)
    assert None not in event.tags
    assert event.tags[:4] == ["geopolitical", "whitehouse", "statement", "other"]
