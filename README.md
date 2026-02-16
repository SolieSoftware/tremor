# Tremor

Event-driven causal shock monitor for financial markets.

## Overview

Tremor lets you register economic and financial events (Fed announcements, earnings releases, geopolitical shocks, economic data releases), transform them into quantified signals via a configurable signal factory, detect when signals constitute shocks, and monitor whether those shocks propagate through a causal network of financial variables as predicted by Granger causality and Structural VAR analysis.

## Architecture

```
                    ┌─────────────┐
                    │   Events    │  Fed announcements, earnings,
                    │   (API)     │  geopolitical, economic data
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Signal    │  Configurable transforms:
                    │   Factory   │  expression evaluation against
                    │             │  event raw_data
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    Shock    │  Z-score based detection
                    │  Detector   │  (threshold_sd configurable
                    │             │   per transform)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Causal    │  Granger causality graph
                    │   Network   │  (networkx DiGraph)
                    │             │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Propagation │  Monitor downstream nodes
                    │   Tracker   │  via market data (yfinance)
                    └─────────────┘
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

### Seed Default Transforms

```bash
python scripts/seed_transforms.py
```

### Run the Server

```bash
uvicorn tremor.app:app --reload
```

API docs available at `http://localhost:8000/docs`.

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

## Development

```bash
# Run tests
pytest

# Run with auto-reload
uvicorn tremor.app:app --reload
```

## Background

Tremor builds on causal analysis of financial market variables using Granger causality testing and Structural VAR models (impulse response functions and forecast error variance decomposition). The causal network and IRF baselines are derived from historical weekly data and loaded at startup from the `data/` directory.
