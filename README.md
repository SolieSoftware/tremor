# Tremor

Event-driven causal shock monitor for financial markets.

## Overview

Tremor lets you register economic and financial events (Fed announcements, earnings releases, geopolitical shocks, economic data releases), transform them into quantified signals via a configurable signal factory, detect when signals constitute shocks, and monitor whether those shocks propagate through a causal network of financial variables as predicted by Granger causality and Structural VAR analysis.

Events can be entered manually via the API, ingested automatically from structured data sources (FRED, Polygon.io), or scraped from unstructured web sources (Federal Reserve press releases, news RSS feeds, White House briefings) using an LLM-based extractor.

## Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │                     Ingestion Layer                      │
  │  FRED API    Polygon.io    Fed Scraper    RSS / WH        │
  │      └──────────────────────┬────────────────────┘       │
  │                       EventPayload                       │
  │                       Normaliser                         │
  └──────────────────────────┬───────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │     Events      │  Fed announcements, earnings,
                    │     (API)       │  geopolitical, economic data
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │     Signal      │  Configurable transforms:
                    │     Factory     │  expression evaluation against
                    │                 │  event raw_data
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │      Shock      │  Z-score based detection
                    │    Detector     │  (threshold_sd configurable
                    │                 │   per transform)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐  ┌───▼────┐  ┌──────▼──────┐
     │    Causal       │  │Causal  │  │ Propagation │
     │  Event Study   │  │Network │  │   Tracker   │
     │  (OLS / plac.) │  │        │  │(market data)│
     └────────────────┘  └────────┘  └─────────────┘
```

**Nodes in the causal network:**
- `d_fed_funds` — Federal funds rate (weekly change)
- `d_treasury_10y` — 10-year Treasury yield (weekly change)
- `d_credit_spread` — High-yield credit spread (weekly change)
- `d_vix` — VIX volatility index (weekly change)
- `sp500_ret` — S&P 500 weekly log returns

## Quick Start

### Install

```bash
git clone <repo-url> && cd tremor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Configure API Keys (optional)

Set via environment variables (prefix `TREMOR_`):

```bash
export TREMOR_FRED_API_KEY=your_key        # free: fred.stlouisfed.org
export TREMOR_POLYGON_API_KEY=your_key     # free: polygon.io/dashboard
export TREMOR_ANTHROPIC_API_KEY=your_key   # for LLM-based scrapers
```

Or create `tremor/.env`:
```
TREMOR_FRED_API_KEY=...
TREMOR_POLYGON_API_KEY=...
TREMOR_ANTHROPIC_API_KEY=...
```

### Seed Default Transforms

```bash
python scripts/seed_transforms.py
```

### Run the Server

```bash
uvicorn tremor.app:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Ingestion

Tremor can pull events from multiple sources automatically.

### FRED (economic data releases)

Requires `TREMOR_FRED_API_KEY`. Supports CPI, NFP, and GDP series.

```bash
# Ingest CPI releases (last 12 months) and compute signals
python scripts/ingest.py fred --series CPIAUCSL --limit 12 --compute-signals

# Ingest NFP and GDP releases
python scripts/ingest.py fred --series PAYEMS A191RL1Q225SBEA --limit 8

# Preview without writing
python scripts/ingest.py fred --series CPIAUCSL --dry-run
```

### Polygon.io (earnings)

Requires `TREMOR_POLYGON_API_KEY`. Fetches actual vs. expected EPS.

```bash
python scripts/ingest.py polygon --ticker AAPL --limit 8 --compute-signals
python scripts/ingest.py polygon --ticker MSFT --limit 4
```

### Federal Reserve scraper (FOMC press releases)

Requires `TREMOR_ANTHROPIC_API_KEY`. Scrapes federalreserve.gov and extracts fields via Claude.

```bash
python scripts/ingest.py fed --limit 5 --compute-signals
```

### RSS news feeds (geopolitical / macro)

Requires `TREMOR_ANTHROPIC_API_KEY`. Scrapes Reuters and AP feeds.

```bash
python scripts/ingest.py rss --limit 10 --compute-signals
```

### White House briefings

Requires `TREMOR_ANTHROPIC_API_KEY`.

```bash
python scripts/ingest.py whitehouse --limit 5
```

---

## API Usage

### 1. Register a Signal Transform

```bash
curl -X POST http://localhost:8000/signals/transforms \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Fed Rate Surprise",
    "event_types": ["fed_announcement"],
    "transform_expression": "actual_rate - expected_rate",
    "node_mapping": "d_fed_funds",
    "unit": "percent",
    "threshold_sd": 2.0
  }'
```

### 2. Register an Event

```bash
curl -X POST http://localhost:8000/events \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-12-18T14:00:00Z",
    "type": "fed_announcement",
    "subtype": "rate_decision",
    "description": "FOMC cuts rates by 25bps, hawkish guidance",
    "tags": ["fomc", "rate_cut"],
    "raw_data": {
      "expected_rate": 4.375,
      "actual_rate": 4.375
    }
  }'
```

### 3. Compute Signals

```bash
curl -X POST http://localhost:8000/signals/compute/{event_id}
```

Returns computed signals with z-scores and shock flags.

### 4. Check for Shocks

```bash
curl http://localhost:8000/monitor/shocks
```

### 5. Monitor Propagation

```bash
# View propagation predictions for a shock
curl http://localhost:8000/monitor/shocks/{signal_id}/propagation

# Trigger a propagation check (pulls latest market data)
curl -X POST http://localhost:8000/monitor/shocks/{signal_id}/check
```

### 6. View Causal Network

```bash
curl http://localhost:8000/monitor/network
```

### 7. Run Causal Event Study

```bash
curl -X POST http://localhost:8000/causal-tests/run \
  -H "Content-Type: application/json" \
  -d '{
    "transform_id": "<uuid>",
    "target_node": "d_treasury_10y",
    "pre_window_days": 5,
    "post_window_days": 5
  }'

# List past results
curl http://localhost:8000/causal-tests

# Check feasibility (how many signals per transform)
curl http://localhost:8000/causal-tests/feasibility
```

---

## Causal Event Study Pipeline

Once enough signals have been ingested (min 5 per transform), run the causal pipeline to test whether event surprises cause statistically significant market responses.

```bash
# Check which transforms have enough data
python scripts/causal_pipeline.py --feasibility

# List available transforms
python scripts/causal_pipeline.py --list-transforms

# Run the full pipeline for a transform
python scripts/causal_pipeline.py --transform "CPI Surprise"

# Custom windows and significance level
python scripts/causal_pipeline.py \
  --transform "CPI Surprise" \
  --pre-window 5 \
  --post-window 10 \
  --significance 0.05

# Test against a specific target node only
python scripts/causal_pipeline.py --transform "CPI Surprise" --target sp500_ret
```

The pipeline runs OLS dose-response regression (post-event return ~ surprise magnitude) with HC1 robust standard errors, plus two placebo tests:
- **Pre-drift placebo**: checks for information leakage before the event
- **Zero-surprise placebo**: verifies that near-zero surprises produce no market response

---

## Database CLI

Inspect the SQLite database without writing SQL:

```bash
python scripts/db_cli.py status              # summary counts for all tables
python scripts/db_cli.py events              # recent events
python scripts/db_cli.py events --type economic_data
python scripts/db_cli.py transforms          # registered signal transforms
python scripts/db_cli.py signals             # computed signals
python scripts/db_cli.py signals --shock     # shocks only
python scripts/db_cli.py propagation         # propagation results
python scripts/db_cli.py causal              # causal test results
python scripts/db_cli.py event <id-prefix>   # full detail for one event
python scripts/db_cli.py shell               # drop into sqlite3 shell
```

---

## Development

```bash
# Run tests
pytest

# Run with auto-reload
uvicorn tremor.app:app --reload
```

---

## Background

Tremor builds on causal analysis of financial market variables using Granger causality testing and Structural VAR models (impulse response functions and forecast error variance decomposition). The causal network and IRF baselines are derived from historical weekly data and loaded at startup from the `data/` directory.

The causal event study module provides a complementary, event-driven approach: OLS dose-response regressions test whether the magnitude of an economic surprise predicts the size of the market response, with placebo tests guarding against false positives.
