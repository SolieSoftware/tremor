# Tremor — Event Data Sources

This document defines every external data source used to populate Tremor events. For each source it specifies:
- What event type it produces
- Whether ingestion is **structured** (API/CSV → direct field mapping) or **unstructured** (scrape → LLM summary → fields)
- The canonical `raw_data` fields that downstream signal transforms expect
- The field that holds the AI summary for unstructured sources (`summary_text`)

---

## Canonical `raw_data` Field Registry

All events share a common field vocabulary in `raw_data`. Structured sources populate these
directly; unstructured sources ask the LLM to extract them where possible, and fall back to
`null` if they cannot be determined from the text.

| Field | Type | Description | Used by transforms |
|---|---|---|---|
| `actual_rate` | float | Actual policy rate (%) | Fed Rate Surprise |
| `expected_rate` | float | Market-implied expected rate (%) | Fed Rate Surprise |
| `actual_cpi` | float | Actual CPI reading (%) | CPI Surprise |
| `expected_cpi` | float | Consensus forecast CPI (%) | CPI Surprise |
| `actual_nfp` | float | Actual non-farm payrolls (thousands) | NFP Surprise |
| `expected_nfp` | float | Consensus forecast NFP (thousands) | NFP Surprise |
| `actual_gdp` | float | Actual GDP growth rate (%) | GDP Surprise |
| `expected_gdp` | float | Consensus forecast GDP (%) | GDP Surprise |
| `actual_eps` | float | Actual earnings per share ($) | Earnings Beat |
| `expected_eps` | float | Analyst consensus EPS ($) | Earnings Beat |
| `vix_before` | float | VIX index level at event open | VIX Spike |
| `vix_after` | float | VIX index level after event | VIX Spike |
| `spread_before` | float | HY credit spread (bps) before event | Credit Stress |
| `spread_after` | float | HY credit spread (bps) after event | Credit Stress |
| `yield_before` | float | 10Y Treasury yield (%) before event | Treasury Yield Shock |
| `yield_after` | float | 10Y Treasury yield (%) after event | Treasury Yield Shock |
| `summary_text` | str | Free-text AI summary (all sources) | — |
| `source_url` | str | URL the data was fetched from | — |
| `source_name` | str | Human-readable source name | — |

---

## Source Catalogue

---

### 1. Federal Reserve — Rate Decisions

| Property | Value |
|---|---|
| **Event type** | `fed_announcement` |
| **Subtype** | `rate_decision` |
| **Ingestion method** | Structured + scrape |
| **URL** | https://www.federalreserve.gov/monetarypolicy/fomccalendar.htm |
| **Format** | HTML press release |
| **Cadence** | ~8× per year (FOMC schedule) |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_rate` | Fed press release | Structured parse (regex for rate range → midpoint) |
| `expected_rate` | CME FedWatch (see source 2) | API |
| `summary_text` | Full press release text | LLM summary |

**Notes:** The press release gives the actual decision. The `expected_rate` must be fetched
from CME FedWatch *before* the announcement (30-day fed funds futures implied rate). These
two sources must be joined on the announcement date.

---

### 2. CME FedWatch — Expected Rate (Fed Futures)

| Property | Value |
|---|---|
| **Event type** | Supplement to `fed_announcement` |
| **Ingestion method** | Structured (web scrape of CME data page) |
| **URL** | https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html |
| **Format** | Embedded JSON / JavaScript data in HTML |
| **Cadence** | Fetched day-of for each FOMC meeting |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `expected_rate` | Implied probability distribution over rate outcomes | Parse embedded JSON, compute probability-weighted rate |

---

### 3. Federal Reserve — Other Announcements

| Property | Value |
|---|---|
| **Event type** | `fed_announcement` |
| **Subtype** | `speech`, `minutes`, `testimony` |
| **Ingestion method** | Unstructured (scrape → LLM) |
| **URL** | https://www.federalreserve.gov/newsevents/speeches.htm |
| **Format** | HTML |
| **Cadence** | Irregular |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `summary_text` | Full speech/minutes text | LLM summary (see LLM schema below) |
| `yield_before` / `yield_after` | FRED DGS10 fetched around event date | Structured API |
| `vix_before` / `vix_after` | Yahoo ^VIX fetched around event date | Structured API |

**LLM extraction schema:**
```json
{
  "summary_text": "2-3 sentence summary of the key policy signals",
  "tone": "hawkish | dovish | neutral",
  "key_topics": ["inflation", "employment", ...],
  "rate_bias": "hike | hold | cut | unclear",
  "actual_rate": null,
  "expected_rate": null
}
```

---

### 4. Bureau of Labor Statistics — CPI

| Property | Value |
|---|---|
| **Event type** | `economic_data` |
| **Subtype** | `cpi_release` |
| **Ingestion method** | Structured (FRED API) |
| **FRED series** | `CPIAUCSL` (monthly CPI, all urban consumers) |
| **Consensus source** | Trading Economics / Econoday scrape |
| **Cadence** | Monthly |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_cpi` | FRED `CPIAUCSL` | FRED API → YoY % change computed |
| `expected_cpi` | Trading Economics calendar | HTML scrape |
| `summary_text` | BLS press release (bls.gov/news.release/cpi) | LLM summary |

---

### 5. Bureau of Labor Statistics — Non-Farm Payrolls

| Property | Value |
|---|---|
| **Event type** | `economic_data` |
| **Subtype** | `nfp_release` |
| **Ingestion method** | Structured (FRED API) |
| **FRED series** | `PAYEMS` (total nonfarm employees, thousands) |
| **Consensus source** | Trading Economics / Econoday scrape |
| **Cadence** | Monthly (first Friday) |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_nfp` | FRED `PAYEMS` (MoM change) | FRED API → diff |
| `expected_nfp` | Trading Economics calendar | HTML scrape |
| `summary_text` | BLS Employment Situation press release | LLM summary |

---

### 6. Bureau of Economic Analysis — GDP

| Property | Value |
|---|---|
| **Event type** | `economic_data` |
| **Subtype** | `gdp_release` |
| **Ingestion method** | Structured (FRED API) |
| **FRED series** | `A191RL1Q225SBEA` (real GDP, % change, quarterly) |
| **Consensus source** | Trading Economics / Econoday scrape |
| **Cadence** | Quarterly (advance, second, third estimates) |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_gdp` | FRED `A191RL1Q225SBEA` | FRED API |
| `expected_gdp` | Trading Economics calendar | HTML scrape |
| `summary_text` | BEA press release (bea.gov/news) | LLM summary |

---

### 7. Earnings Releases — Public Companies

| Property | Value |
|---|---|
| **Event type** | `earnings` |
| **Subtype** | `eps_release` |
| **Ingestion method** | Structured (Polygon.io API) |
| **API** | Polygon.io `/v2/reference/financials` + `/vX/reference/financials` |
| **Cadence** | Quarterly (earnings season) |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_eps` | Polygon.io earnings endpoint | Structured API |
| `expected_eps` | Polygon.io analyst estimates endpoint | Structured API |
| `summary_text` | SEC EDGAR 8-K press release text | LLM summary |

**Notes:** Polygon.io free tier covers delayed data. For real-time earnings, upgrade to paid
or supplement with Yahoo Finance earnings calendar (embedded JSON in page HTML).

---

### 8. SEC EDGAR — 8-K Filings (Material Events)

| Property | Value |
|---|---|
| **Event type** | `earnings` or `fed_announcement` (bank-specific) |
| **Subtype** | `8k_filing` |
| **Ingestion method** | Unstructured (EDGAR full-text search → LLM) |
| **API** | https://efts.sec.gov/LATEST/search-index?q=&dateRange=custom |
| **Format** | HTML / XBRL |
| **Cadence** | Continuous |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `actual_eps` | 8-K earnings press release | LLM extraction |
| `expected_eps` | Analyst consensus (if mentioned in filing) | LLM extraction |
| `summary_text` | Full 8-K text | LLM summary |

**LLM extraction schema:**
```json
{
  "summary_text": "2-3 sentence summary of the material event",
  "event_category": "earnings | guidance | merger | regulatory | other",
  "actual_eps": null,
  "expected_eps": null,
  "revenue_actual": null,
  "revenue_expected": null,
  "guidance_direction": "raised | lowered | maintained | none"
}
```

---

### 9. Reuters / AP RSS Feeds — Geopolitical Events

| Property | Value |
|---|---|
| **Event type** | `geopolitical` |
| **Subtype** | `news_event` |
| **Ingestion method** | Unstructured (RSS → Playwright scrape → LLM) |
| **Feeds** | Reuters: https://feeds.reuters.com/reuters/topNews, AP: https://rsshub.app/apnews/topics/ap-top-news |
| **Format** | RSS XML → article HTML |
| **Cadence** | Continuous polling (configurable interval) |

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `vix_before` / `vix_after` | Yahoo ^VIX fetched ±1 day around event | yfinance |
| `spread_before` / `spread_after` | FRED BAMLH0A0HYM2 ±1 day | FRED API |
| `summary_text` | Article full text | LLM summary |

**LLM extraction schema:**
```json
{
  "summary_text": "2-3 sentence factual summary of the event",
  "event_category": "conflict | sanctions | election | policy | natural_disaster | other",
  "countries_involved": ["US", "China", ...],
  "severity": "low | medium | high",
  "market_relevance": "fx | rates | equities | commodities | broad",
  "vix_before": null,
  "vix_after": null,
  "spread_before": null,
  "spread_after": null
}
```

---

### 10. White House — Presidential Announcements

| Property | Value |
|---|---|
| **Event type** | `geopolitical` |
| **Subtype** | `presidential_statement` |
| **Ingestion method** | Unstructured (scrape → LLM) |
| **URL** | https://www.whitehouse.gov/briefings-statements/ |
| **Format** | HTML |
| **Cadence** | Irregular, poll daily |

**Fields populated:** Same as Reuters/AP schema above, plus:

```json
{
  "policy_area": "trade | defense | fiscal | monetary | regulatory | other",
  "executive_action": true,
  "affected_sectors": ["technology", "energy", ...]
}
```

---

### 11. European Central Bank — Rate Decisions & Statements

| Property | Value |
|---|---|
| **Event type** | `fed_announcement` (use `source_name` = "ECB") |
| **Subtype** | `rate_decision` or `speech` |
| **Ingestion method** | Structured (rate) + Unstructured (statement text) |
| **URL** | https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html |
| **Format** | HTML |
| **Cadence** | ~8× per year |

**Fields populated:** Same as Federal Reserve sources (1 & 3), with `source_name = "ECB"`.

---

### 12. Bank of England — MPC Decisions

| Property | Value |
|---|---|
| **Event type** | `fed_announcement` (use `source_name` = "BoE") |
| **Subtype** | `rate_decision` |
| **Ingestion method** | Structured (rate parse) + Unstructured (statement) |
| **URL** | https://www.bankofengland.co.uk/monetary-policy/the-interest-rate-bank-rate |
| **Format** | HTML |
| **Cadence** | ~8× per year |

---

### 13. GDELT Project — Global Events Database

| Property | Value |
|---|---|
| **Event type** | `geopolitical` |
| **Subtype** | `gdelt_event` |
| **Ingestion method** | Structured (bulk CSV download) |
| **URL** | https://www.gdeltproject.org/data.html |
| **Format** | CSV (tab-delimited) |
| **Cadence** | 15-minute updates |

**Notes:** GDELT pre-codes events using the CAMEO taxonomy (Conflict and Mediation Event
Observations). This is an alternative to scraping news sites — useful for systematic
backtesting over historical event windows. Not a real-time feed but excellent for batch
ingestion of historical geopolitical events.

**Fields populated:**

| Field | Source | Method |
|---|---|---|
| `summary_text` | GDELT event description + linked article | LLM summary of source article |
| `vix_before` / `vix_after` | Yahoo ^VIX at event date | yfinance |

---

## Ingestion Strategy Summary

| Category | Sources | Method | Expected/Actual split |
|---|---|---|---|
| Fed rate decisions | Fed.gov + CME FedWatch | Structured | Both available |
| Fed speeches/minutes | Fed.gov | Unstructured → LLM | Neither (market proxy) |
| CPI / NFP / GDP | FRED API | Structured | Actual from FRED; expected from consensus scrape |
| Earnings | Polygon.io + EDGAR | Structured + LLM | Both via Polygon |
| Geopolitical (news) | Reuters/AP RSS | Unstructured → LLM | Neither (market proxy) |
| Presidential | White House | Unstructured → LLM | Neither (market proxy) |
| Central banks (ECB/BoE) | Official sites | Structured + LLM | Rate from site; expected from market |
| Historical bulk | GDELT | Structured CSV | Market proxy only |

---

## Module Design

The ingestion layer lives at `tremor/ingestion/` and is structured as follows:

```
tremor/ingestion/
├── __init__.py
├── base.py              # BaseIngester ABC and EventPayload dataclass
├── normaliser.py        # Converts any ingester output → EventCreate schema
│
├── api/                 # Structured sources (direct field mapping)
│   ├── __init__.py
│   ├── fred.py          # FRED API → CPI, NFP, GDP actuals
│   ├── polygon.py       # Polygon.io → EPS actual/expected
│   └── cme_fedwatch.py  # CME FedWatch → expected Fed rate
│
└── scrapers/            # Unstructured sources (Playwright + LLM)
    ├── __init__.py
    ├── browser.py        # Copied/adapted from smart-webscraper-products
    ├── llm_extractor.py  # Anthropic Claude extraction with per-source schemas
    ├── fed_scraper.py    # Fed.gov press releases
    ├── rss_scraper.py    # Reuters/AP RSS → article scrape
    └── whitehouse_scraper.py  # White House briefings
```

### Pattern: Structured sources

```python
# api/fred.py
class FredIngester(BaseIngester):
    async def fetch(self, series_id: str, ...) -> list[EventPayload]:
        # Call FRED API, map fields directly to raw_data
        ...
```

### Pattern: Unstructured sources

```python
# scrapers/fed_scraper.py
class FedScraper(BaseIngester):
    async def fetch(self, url: str) -> list[EventPayload]:
        # 1. Playwright: fetch HTML
        # 2. BeautifulSoup: strip noise
        # 3. Claude: extract fields per LLM schema above
        # 4. Normaliser: fill any missing numeric fields with None
        ...
```

### Shared LLM extraction

All scrapers call `llm_extractor.extract(html, schema)` which:
1. Cleans HTML with BeautifulSoup (strips scripts/styles, truncates to 50k chars)
2. Sends to Claude with the per-source JSON schema as the target format
3. Validates output with Pydantic
4. Returns a dict that the normaliser maps into `raw_data`

The LLM is used with `temperature=0.0` for determinism. The model is `claude-sonnet-4-6`.
