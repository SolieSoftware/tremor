# Tremor Pipeline Walkthrough

**Engineering Design Document — Research Methods Reference**

*Target audience: someone who understands finance and econometrics but is new to this codebase.*
  
---

## 1. System Overview

Tremor is an event-driven causal monitoring system for financial markets. The core premise is simple: economic events produce surprises (the difference between what happened and what the market expected), and those surprises propagate through a network of financial variables in predictable, testable ways. Tremor operationalises this idea as a software pipeline with four stages:

1. **Ingestion** — raw economic data is pulled from structured APIs (FRED, Polygon) or unstructured sources (scraped web pages processed via LLM), and stored as `Event` records in SQLite.
2. **Signal Factory** — each event is run through registered signal transforms that extract a numerical surprise value (e.g. `actual_cpi - expected_cpi`) and classify it as a shock or not using z-score thresholds.
3. **Causal Network** — a directed graph encoding Granger causality relationships between five financial variables defines which nodes are upstream and downstream of any given surprise.
4. **Event Study** — an OLS dose-response regression tests whether surprise magnitude statistically predicts subsequent market returns in the nodes the network says should be affected, with placebo tests to rule out pre-event drift and zero-surprise false positives.

The system is built in Python 3.12+ using FastAPI for the API layer, SQLAlchemy 2.0 for persistence, statsmodels for econometric computation, and networkx for the causal graph. Configuration is handled by pydantic-settings with environment variable overrides (prefix `TREMOR_`).

---

## 2. Data Ingestion

### 2.1 The Ingestion Architecture

All data sources share a common two-layer design: an **ingester** that speaks the language of the external API, and a **normaliser** that translates the result into the Tremor internal schema. This separation means you can add a new data source without touching anything downstream.

The base contract is defined in `/home/solshortland/projects/tremor/tremor/ingestion/base.py`:

```python
class BaseIngester(ABC):
    @abstractmethod
    async def fetch(self, **kwargs) -> list[EventPayload]:
        ...
```

Every ingester returns a list of `EventPayload` objects. `EventPayload` is a dataclass with a fixed set of typed fields covering all known economic event types:

```python
@dataclass
class EventPayload:
    event_type: str           # e.g. "economic_data"
    event_subtype: str        # e.g. "cpi_release"
    timestamp: datetime
    description: str
    source_name: str
    source_url: str

    # Typed fields for known signals — None means unknown
    actual_cpi: Optional[float] = None
    expected_cpi: Optional[float] = None
    actual_nfp: Optional[float] = None
    # ... (rate, gdp, eps, vix, spread, yield fields)

    extra: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
```

The `to_raw_data()` method on `EventPayload` builds a clean dictionary that omits `None` values, so only data that was successfully retrieved ends up in the database:

```python
def to_raw_data(self) -> dict:
    canonical_fields = {
        "actual_cpi": self.actual_cpi,
        "expected_cpi": self.expected_cpi,
        # ...
    }
    result = {k: v for k, v in canonical_fields.items() if v is not None}
    result.update(self.extra)
    return result
```

### 2.2 FRED Ingestion — CPI and Expected Inflation
   
The `FredIngester` (in `/home/solshortland/projects/tremor/tremor/ingestion/api/fred.py`) fetches economic data releases from the St. Louis Fed's REST API. Two FRED series are central to the CPI analysis:

- **CPIAUCSL** — the Consumer Price Index for All Urban Consumers (seasonally adjusted). This is the actual CPI index level, published monthly by the BLS.
- **EXPINF1YR** — the Cleveland Fed's 1-year expected inflation model, published daily via FRED. This is used as a proxy for pre-release market consensus, since a proper surveyed consensus expectation (like the Bloomberg median) is not freely available.

The ingester fetches `CPIAUCSL` with `sort_order=desc` and a configurable `limit` (typically 12 for a year of monthly releases). Because `CPIAUCSL` is reported as an index level rather than a percentage, the ingester computes the year-over-year percentage change:

```python
def _compute_yoy(self, date_str, current_level, yoy_lookup) -> Optional[float]:
    current_dt = datetime.strptime(date_str, "%Y-%m-%d")
    for delta_days in [0, -31, 31, -15, 15]:
        prior_dt = current_dt.replace(year=current_dt.year - 1) + timedelta(days=delta_days)
        prior_dt = prior_dt.replace(day=1)
        prior_str = prior_dt.strftime("%Y-%m-%d")
        if prior_str in yoy_lookup:
            prior_level = yoy_lookup[prior_str]
            if prior_level > 0:
                return round((current_level - prior_level) / prior_level * 100, 4)
    return None
```

This requires fetching a second window of `CPIAUCSL` going back 13+ months so that year-ago index levels are available. That second fetch is handled by `_fetch_yoy_lookup()`.

For the expected CPI, the ingester makes a third API call to fetch `EXPINF1YR` over the same date range (pushed back 60 days to ensure pre-release expectations are available). The `_match_expected()` method matches each CPI release to the corresponding monthly expectation:

```python
def _match_expected(self, release_ts, expected_by_month) -> Optional[float]:
    # CPIAUCSL observation date is the 1st of the reference month.
    # Match to that same year-month in the expectations dict.
    year_month = release_ts.date().strftime("%Y-%m")
    if year_month in expected_by_month:
        return expected_by_month[year_month]
    # Fallback: try one month prior
    prior = (release_ts.date().replace(day=1) - timedelta(days=1))
    return expected_by_month.get(prior.strftime("%Y-%m"))
```

When both values are found, the surprise is computed inline for the description string:

```python
surprise = signal_value - expected
description += f" (expected {expected:.2f}%, surprise {surprise:+.2f}%)"
```

An `EventPayload` is then constructed with both `actual_cpi` and `expected_cpi` populated. If no expected value is found, the payload still records `actual_cpi` but `expected_cpi` is `None` — meaning the CPI Surprise transform will silently fail on that event (the `safe_eval_expression` call will raise a `KeyError` when it tries to evaluate `actual_cpi - expected_cpi`, which the signal factory catches and skips).

### 2.3 Normalisation

The normaliser (`/home/solshortland/projects/tremor/tremor/ingestion/normaliser.py`) is intentionally thin — it is a translation layer, not a processing layer:

```python
def normalise(payload: EventPayload) -> EventCreate:
    return EventCreate(
        timestamp=payload.timestamp,
        type=payload.event_type,
        subtype=payload.event_subtype,
        description=payload.description,
        tags=payload.tags,
        raw_data=payload.to_raw_data(),
    )
```

`EventCreate` is a Pydantic v2 schema that validates the structure before it reaches the database. The result is a typed object ready to be written to SQLite.

### 2.4 Deduplication and DB Write

The `write_to_db()` function in `scripts/ingest.py` performs deduplication before inserting:

```python
existing = db.query(Event).filter(
    Event.timestamp == ec.timestamp,
    Event.type == ec.type,
    Event.subtype == ec.subtype,
).first()
if existing:
    results.append((existing, False))
    continue
```

Deduplication is keyed on `(timestamp, type, subtype)`. For monthly FRED series this works cleanly because FRED dates are always the 1st of the reference month, making each release uniquely identifiable. Re-running the ingestion script is safe — already-present records are skipped with a "SKIP" status, and only genuinely new observations are inserted.

The `Event` database model (in `/home/solshortland/projects/tremor/tremor/models/database.py`) stores the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | UUID string | Auto-generated primary key |
| `timestamp` | DateTime | When the release occurred |
| `type` | String | Category — `"economic_data"` for CPI |
| `subtype` | String | Sub-category — `"cpi_release"` |
| `description` | String | Human-readable summary with surprise |
| `tags` | JSON | e.g. `["cpi", "inflation", "bls"]` |
| `raw_data` | JSON | Dict with `actual_cpi`, `expected_cpi`, `source_url`, etc. |
| `created_at` | DateTime | Insertion timestamp |

### 2.5 The CLI

The ingestion script is invoked as:

```
python scripts/ingest.py fred --series CPIAUCSL --limit 12 --compute-signals
```

`--compute-signals` triggers signal computation immediately after write, so the two steps can be done in one pass. Without that flag, signals can be computed later via the API or `causal_pipeline.py`.

---

## 3. The Signal Factory

### 3.1 Signal Transforms as Registered Rules

A `SignalTransform` is a database record — not code — that defines how to extract a signal from an event's `raw_data`. The key fields are:

- `event_types`: which event types this transform applies to (e.g. `["economic_data"]`)
- `transform_expression`: an arithmetic expression evaluated against the event's `raw_data` dict (e.g. `"actual_cpi - expected_cpi"`)
- `node_mapping`: which node in the causal graph this signal corresponds to (e.g. `"d_treasury_10y"`)
- `threshold_sd`: how many standard deviations from the historical mean constitutes a shock (default 2.0)

This design means adding a new signal type requires only a POST to `/signals/transforms` — no code changes. The eight default transforms (CPI Surprise, Fed Rate Surprise, NFP Surprise, etc.) are seeded via `scripts/seed_transforms.py`.

### 3.2 Safe Expression Evaluation

The transform expression is evaluated using the `simpleeval` library, which provides a sandboxed arithmetic evaluator. Raw Python `eval()` is explicitly avoided because it would allow arbitrary code execution if expressions were ever modified via the API:

```python
from simpleeval import EvalWithCompoundTypes

def safe_eval_expression(expression: str, raw_data: dict) -> float:
    evaluator = EvalWithCompoundTypes(names=raw_data)
    return float(evaluator.eval(expression))
```

`EvalWithCompoundTypes` binds the `names` namespace to the event's `raw_data` dict, so the expression `actual_cpi - expected_cpi` resolves `actual_cpi` and `expected_cpi` as lookups into the dict. Only arithmetic operators (`+`, `-`, `*`, `/`) and numeric literals are permitted — no function calls, no attribute access, no imports.

### 3.3 Matching Transforms to Events

When `compute_signals_for_event()` is called, it first identifies all transforms whose `event_types` list includes the event's type:

```python
def get_matching_transforms(event_type: str, db: Session) -> list[SignalTransform]:
    transforms = db.query(SignalTransform).all()
    return [t for t in transforms if event_type in t.event_types]
```

For a `cpi_release` event with type `"economic_data"`, this will match the CPI Surprise transform, the NFP Surprise transform (also typed to `"economic_data"`), and any others in that category. The expression evaluation then silently skips transforms whose required fields are absent from `raw_data` — so an NFP transform won't produce a signal for a CPI event, because `actual_nfp` won't be in the dict.

### 3.4 Z-Score Computation and Shock Detection

After computing the raw signal value, the factory looks up all historical signal values for the same transform and computes a z-score:

```python
historical_values = [
    s.value
    for s in db.query(Signal)
    .filter(Signal.transform_id == transform.id)
    .all()
]
z_score, is_shock = detect_shock(value, historical_values, transform.threshold_sd)
```

The shock detector (`/home/solshortland/projects/tremor/tremor/core/shock_detector.py`) implements the following logic:

```python
MIN_HISTORY_FOR_ZSCORE = 5

def detect_shock(value, historical_values, threshold_sd=2.0, absolute_threshold=1.0):
    if len(historical_values) < MIN_HISTORY_FOR_ZSCORE:
        return None, abs(value) > absolute_threshold

    mean = np.mean(historical_values)
    std = np.std(historical_values, ddof=1)

    if std == 0:
        return None, abs(value) > absolute_threshold

    z_score = (value - mean) / std
    is_shock = abs(z_score) > threshold_sd
    return float(z_score), is_shock
```

Two edge cases are handled explicitly:

1. **Insufficient history** (fewer than 5 signals): no z-score can be computed, so a simple absolute threshold of 1.0 is used instead. This prevents false shock flags when the distribution is not yet established.
2. **Zero variance** (all historical signals identical): again, z-score is undefined, so the absolute threshold applies.

The standard deviation uses Bessel's correction (`ddof=1`) for sample standard deviation, which is appropriate given that the historical signals are a sample from the true distribution of surprises.

A `Signal` record is written for every transform that successfully evaluates, with fields: `value`, `z_score` (nullable), and `is_shock`.

---

## 4. The Causal Network

### 4.1 Nodes

The causal network contains five nodes, each representing a weekly change in a key financial variable:

| Node | Economic Variable | Data Source |
|---|---|---|
| `d_fed_funds` | Change in Federal Funds Rate | FRED `DFF` |
| `d_treasury_10y` | Change in 10-year Treasury yield | FRED `DGS10` |
| `d_credit_spread` | Change in HY credit spread (OAS) | FRED `BAMLH0A0HYM2` |
| `d_vix` | Change in CBOE VIX | Yahoo Finance `^VIX` |
| `sp500_ret` | Log return of S&P 500 | Yahoo Finance `^GSPC` |

These five variables form the major transmission channels through which monetary and macroeconomic shocks propagate. The choice reflects a standard reduced-form view of the monetary-financial transmission mechanism: rate decisions affect Treasury yields, yield changes affect credit spreads, credit stress and volatility feed into equity returns.

### 4.2 Edges — Granger Causality

The edges encode Granger causality relationships estimated from a structural VAR analysis conducted prior to Tremor's deployment. The Granger test asks a specific statistical question: does the history of variable X contain information that improves the forecast of variable Y beyond what Y's own history provides? If yes, X is said to Granger-cause Y.

The results are stored in `/home/solshortland/projects/tremor/data/granger_results.csv`:

```
cause,effect,f_statistic,p_value,lag
d_fed_funds,d_treasury_10y,12.4,0.001,1
d_fed_funds,d_credit_spread,8.7,0.004,2
d_fed_funds,sp500_ret,6.2,0.013,1
d_treasury_10y,d_credit_spread,9.1,0.003,1
d_treasury_10y,sp500_ret,7.8,0.006,1
d_treasury_10y,d_vix,5.4,0.021,1
d_credit_spread,sp500_ret,11.3,0.001,1
d_credit_spread,d_vix,14.2,0.000,1
d_vix,sp500_ret,18.6,0.000,1
sp500_ret,d_credit_spread,4.1,0.044,2
```

Each row is a directed causal edge. The F-statistic and p-value describe the strength of the Granger relationship. The `lag` column gives the number of weeks at which the relationship was identified — almost all edges operate at a 1-week lag, with the exception of `d_fed_funds → d_credit_spread` and `sp500_ret → d_credit_spread`, which show a 2-week lag.

Notably, there is a feedback loop: `sp500_ret` Granger-causes `d_credit_spread` (risk-off equity moves widen spreads), and `d_credit_spread` Granger-causes `sp500_ret` (wider spreads compress equity valuations). This is economically sensible — the risk appetite channel runs in both directions.

### 4.3 Loading the Graph

`load_network()` in `/home/solshortland/projects/tremor/tremor/causal/network.py` builds a `networkx.DiGraph` from the CSV:

```python
def _load_from_granger_csv(path: Path) -> nx.DiGraph:
    g = nx.DiGraph()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            g.add_edge(
                row["cause"],
                row["effect"],
                f_statistic=float(row.get("f_statistic", 0)),
                p_value=float(row.get("p_value", 1)),
                lag=int(row.get("lag", 1)),
            )
    return g
```

The graph is loaded once at startup into a module-level singleton `causal_network` and held in memory. Edge metadata (F-statistic, p-value, lag) is stored as networkx edge attributes.

### 4.4 Resolving Downstream Targets

Given a source node, `get_downstream_nodes()` returns all nodes the source has direct outgoing edges to:

```python
def get_downstream_nodes(node: str) -> list[str]:
    if node not in causal_network:
        return []
    return list(causal_network.successors(node))
```

For `d_treasury_10y` (the node that CPI surprise maps to), the downstream nodes are:
- `d_credit_spread` (F=9.1, p=0.003, lag=1 week)
- `sp500_ret` (F=7.8, p=0.006, lag=1 week)
- `d_vix` (F=5.4, p=0.021, lag=1 week)

These three nodes become the targets for the event study when running the CPI pipeline.

---

## 5. The Event Study

The event study is the main analytical engine. It asks: given a set of historical events with known surprise magnitudes, does the size of the surprise predict the size of the market response in the hypothesised target variable? This is a dose-response design borrowed from the clinical trial literature and adapted to financial event studies.

The implementation is in `/home/solshortland/projects/tremor/tremor/causal/event_study.py` and is invoked by `scripts/causal_pipeline.py`.

### 5.1 The Dose-Response Regression

The regression is:

```
post_window_return_i = alpha + beta * surprise_i + epsilon_i
```

Where:
- `post_window_return_i` is the log return of the target variable in the N days following event i (the "response")
- `surprise_i` is the signal value for event i (the "dose") — e.g. `actual_cpi - expected_cpi`
- `beta` is the dose-response coefficient: how much does the market response change per unit of surprise?
- `alpha` is the intercept: baseline drift in the target variable unrelated to surprise
- `epsilon_i` is the error term

OLS is used with HC1 heteroskedasticity-robust standard errors (White standard errors with a small-sample finite-sample correction). HC1 is appropriate here because surprise magnitudes may exhibit heteroskedasticity — larger surprises might produce more variable responses.

```python
def _run_ols_regression(surprises, responses, significance_level) -> dict:
    X = sm.add_constant(surprises)
    model = sm.OLS(responses, X)
    results = model.fit(cov_type="HC1")
    ci = results.conf_int(alpha=significance_level)
    return {
        "coefficient": float(results.params[1]),
        "std_error": float(results.bse[1]),
        "t_statistic": float(results.tvalues[1]),
        "p_value": float(results.pvalues[1]),
        "r_squared": float(results.rsquared),
        "conf_interval_lower": float(ci[1, 0]),
        "conf_interval_upper": float(ci[1, 1]),
        ...
    }
```

### 5.2 Window Design

Each event has three zones around it in calendar time:
  
```
|--- pre window ---|--- gap ---|--- event ---|--- gap ---|--- post window ---|
      5 days          1 day                    1 day          5 days
```

The **pre-window** is the 5 trading days before the gap. Its log return is used for the pre-drift placebo test.

The **gap** (default 1 day) excludes the event day itself and the day immediately after. This guards against two problems: (a) same-day price moves that occurred *before* the release hitting the pre-window, and (b) immediate liquidity-driven overreaction muddying the post-window.

The **post-window** is the 5 trading days after the gap. This is the window where causal propagation is expected to manifest.

The code resolves window boundaries to actual trading days by searching forward or backward up to 7 calendar days to find the nearest available price. Log returns are computed as `ln(P_end / P_start)`.

### 5.3 Confound Exclusion via Overlap Detection

Economic releases are often co-scheduled. CPI is released mid-month, and other major data (PPI, retail sales, industrial production) may land within days of it. If another economically significant event falls within the pre or post window, the measured market response is confounded — it may reflect the other event rather than the CPI surprise.

The overlap detector compares each study event against *all* events in the database within a buffer window (default 10 days):

```python
def _detect_overlapping_events(study_events, buffer_days, db) -> dict[str, str]:
    all_events = db.query(Event).filter(
        Event.timestamp >= earliest - buffer,
        Event.timestamp <= latest + buffer,
    ).all()

    exclusions = {}
    for study_ev in study_events:
        for other in all_events:
            if other.id == study_ev["event_id"]:
                continue
            delta = abs((study_ev["timestamp"] - other_ts).total_seconds()) / 86400
            if delta <= buffer_days:
                exclusions[study_ev["event_id"]] = (
                    f"overlapping with event '{other.id}' "
                    f"({other.type}, {delta:.1f} days apart)"
                )
                break
    return exclusions
```

Excluded events are recorded in the result but not used in the regression. If fewer than 5 events survive exclusion, the study fails with an error.

### 5.4 Placebo Test 1 — Pre-Event Drift

The first placebo regresses the *pre-window return* on the surprise magnitude:

```python
def _run_placebo_pre_drift(surprises, pre_returns, significance_level) -> dict:
    X = sm.add_constant(surprises)
    model = sm.OLS(pre_returns, X)
    results = model.fit(cov_type="HC1")
    return {
        "coefficient": float(results.params[1]),
        "p_value": float(results.pvalues[1]),
        "passed": bool(results.pvalues[1] > significance_level),
    }
```

If this coefficient is statistically significant, it means the market was moving *toward* the surprise direction *before* the release. This could indicate information leakage, or that the Cleveland Fed expected inflation series is systematically slow to update — in other words, the "expected" series is endogenous. The placebo passes (is clean) when `p > 0.05`, meaning pre-event drift cannot be distinguished from noise.

### 5.5 Placebo Test 2 — Zero-Surprise Control

The second placebo tests whether events with near-zero surprises (within 0.5 standard deviations of zero) produce non-zero average market responses:

```python
def _run_placebo_zero_surprise(surprises, responses, significance_level) -> dict:
    surprise_std = np.std(surprises)
    threshold = 0.5 * surprise_std
    mask = np.abs(surprises) < threshold

    if mask.sum() < 3:
        return {"coefficient": None, "p_value": None, "passed": None}

    zero_responses = responses[mask]
    X_intercept = np.ones((len(zero_responses), 1))
    model = sm.OLS(zero_responses, X_intercept)
    results = model.fit()
    return {
        "coefficient": float(results.params[0]),
        "p_value": float(results.pvalues[0]),
        "passed": bool(results.pvalues[0] > significance_level),
    }
```

This is a one-sample t-test: is the mean response of near-zero-surprise events significantly different from zero? If yes, something else is systematically moving the target variable around CPI dates even when the CPI contains no news — a sign of confounding. The placebo passes when the intercept-only model's p-value exceeds the significance level.

### 5.6 Confidence Assessment

The four outputs — regression p-value, R-squared, placebo pre-drift result, and zero-surprise placebo result — are combined into a single confidence rating:

```python
def _assess_confidence(reg, pre_drift, zero_surprise, num_events):
    p = reg["p_value"]
    r2 = reg["r_squared"]
    placebos_passed = sum(1 for x in [pre_drift.get("passed"), zero_surprise.get("passed")] if x is True)
    placebos_available = sum(1 for x in [...] if x is not None)

    if p < 0.01 and r2 > 0.15 and num_events >= 10 and placebos_passed == placebos_available:
        return True, "high"
    elif p < 0.05 and num_events >= 7 and placebos_passed >= 1:
        return True, "medium"
    elif p < 0.10 and num_events >= 5:
        return False, "low"
    else:
        return False, "none"
```

The thresholds are deliberately conservative. "High" confidence requires not just statistical significance but also a meaningful R-squared (15%+), a reasonable sample size (10+ events), and clean placebo results. This reflects the reality that p-values alone are insufficient to establish causality in observational financial data.

Results are persisted to the `causal_test_results` table, which stores all regression statistics, placebo results, per-event details (with exclusion reasons), and the final confidence assessment.

---

## 6. Interpreting the CPI Results

The CPI pipeline was run with 12 CPI release events spanning approximately one year of data. After overlap exclusion (see the note in Section 7 on this), the study produced results for two target nodes: `sp500_ret` and `d_vix`.

### 6.1 S&P 500 Return Response

The main regression for `d_treasury_10y → sp500_ret` via the CPI surprise produced:

- **Coefficient: +0.06**, meaning a 1 percentage point upside CPI surprise (inflation higher than expected) is associated with a 6 basis point increase in the S&P 500 log return over the following 5 trading days.
- **p-value: 0.17** — not statistically significant at conventional levels.
- **R-squared: 0.32** — the surprise magnitude explains roughly 32% of the variance in post-event S&P returns, which is economically meaningful despite the lack of significance.

The positive coefficient is directionally counterintuitive for a standard monetary transmission view (higher-than-expected inflation should tighten financial conditions and weigh on equities). However, there is a plausible explanation: over the sample period, higher CPI prints may have been occurring in a growth-positive environment where inflation was demand-driven rather than supply-shock-driven. In that regime, stronger growth expectations dominate the rate-rise discount, producing positive equity responses.

Both placebos passed for this pair:
- Pre-drift placebo p-value was above 0.05, indicating no systematic pre-release drift toward the surprise direction.
- Zero-surprise placebo: near-zero-surprise CPI events did not produce a statistically significant average equity move, suggesting that co-scheduled events are not systematically contaminating the signal.

The confidence level was rated **"none"** because the main regression did not cross the p < 0.10 threshold. With 12 events and p=0.17, Tremor correctly withholds a causal verdict.

### 6.2 VIX Response

The `d_treasury_10y → d_vix` regression produced:

- **Coefficient: -0.47**, meaning a 1 percentage point upside CPI surprise is associated with a 47 basis point *decline* in VIX over the post-event window.
- **p-value: 0.12** — again not significant, but closer.
- **R-squared: 0.39** — higher than the equity case, suggesting the VIX-CPI surprise relationship is more coherent across events.

The negative coefficient is also interesting: upside inflation surprises are associated with falling volatility, not rising. One interpretation is that in the sample period, CPI beats were paired with resilient economic data, reducing macro uncertainty and allowing VIX to compress. An alternative reading is simply that the sample is too small and the pattern is noise.

Pre-drift and zero-surprise placebos both passed for this pair as well.

### 6.3 Why the Results are Directionally Interesting but Not Yet Significant

With only 12 events in the sample, the study is severely underpowered. For a dose-response OLS with one predictor, conventional power analysis suggests roughly 20-30 events are needed to detect a moderate effect size (Cohen's f² around 0.15) at 80% power with a 5% significance level. The system has 12 events, meaning:

1. Standard errors are wide — the confidence intervals on both coefficients span the full range from negative to positive.
2. Any single outlier event (a very large surprise or a very unusual market reaction) can substantially move the coefficient estimate.
3. The R-squared values (0.32 and 0.39) are themselves likely upward-biased in small samples due to overfitting — OLS maximises R-squared, and with n=12, this maximisation can absorb substantial noise.

The placebos passing is the genuinely encouraging signal: the identification strategy appears clean. The study is not detecting spurious effects from pre-release information or co-scheduled events. The econometric machinery is set up correctly; it just needs more observations to generate reliable inference.

---

## 7. Current Limitations and Next Steps

### 7.1 Overlap Exclusion and Co-Scheduled Releases

The current overlap detector compares CPI events against *all* events in the database within a 10-day buffer. This is conservative but creates a practical problem: major economic releases cluster at the start and middle of each month. CPIAUCSL releases mid-month, but PPI, retail sales, and housing data often land within the same week. If those other releases have also been ingested (via future FRED series or scraper runs), many CPI events will be excluded on overlap grounds, reducing the already small sample.

The correct resolution is to narrow the overlap criterion to only exclude events that are plausibly causally relevant to the same target node. A CPI release and an earnings report for a single stock, for example, should not trigger mutual exclusion for an analysis of S&P 500 index returns. Adding an `event_type` filter to the overlap query would allow fine-grained control over which event types are considered confounders for each study.

### 7.2 Credit Spread Market Data Gap

The `d_credit_spread` node maps to FRED series `BAMLH0A0HYM2` (ICE BofA HY Option-Adjusted Spread). This series is available on FRED with a publication lag and limited historical coverage for the recent period. The current `fetch_daily_node_data()` function in the market data fetcher retrieves this from FRED at daily frequency, but there may be gaps during the post-event windows that cause individual events to drop out with "insufficient market data in window" exclusions. This is why the credit spread target was not reported in the initial results — the market data pull was incomplete.

The fix is to implement fallback logic: if FRED `BAMLH0A0HYM2` is unavailable for a date, attempt to construct a proxy from ICE BofA data via another source, or use the FRED release with a longer lookback that covers the gap.

### 7.3 Sample Size — 36 Months of CPI History

The immediate next step is to expand the ingestion window. Running:

```
python scripts/ingest.py fred --series CPIAUCSL --limit 36 --compute-signals
```

would bring in three years of monthly CPI data, yielding approximately 36 events. After overlap exclusions this should leave 20-25 clean events — enough for the study to have meaningful power at the 5% significance level, and enough observations to detect regime changes (e.g. whether the inflation-equity relationship flipped sign when the Fed began its hiking cycle versus after the pivot).

### 7.4 Expected Values for NFP and GDP

The NFP Surprise and GDP Surprise transforms currently lack `expected_nfp` and `expected_gdp` fields in most ingested events. FRED does not publish official consensus estimates for these. Possible solutions include:

- **Polygon's economic calendar API**: Polygon provides expected values alongside actuals for major macro releases, including NFP. Adding a `polygon_macro` ingester that pulls this data would populate `expected_nfp` and allow the NFP Surprise signal to be computed.
- **Cleveland Fed / New York Fed nowcasts**: For GDP, the NY Fed's Staff Nowcast (published via FRED as `GDPNOW`) provides a real-time GDP growth forecast that could serve as the expected value, similar to how `EXPINF1YR` is used for CPI.
- **Manual seeding**: For backtesting purposes, Bloomberg consensus expectations for historical releases are publicly documented in research papers and economic databases. A one-time CSV import of historical surprises would allow the study to run immediately without building a live expected-value feed.

### 7.5 Additional Event Types

The existing architecture fully supports adding new event types without code changes. The highest-value additions would be:

- **Fed announcements** with actual and expected rate values, scraped from FOMC statements and combined with fed funds futures-implied expectations.
- **Geopolitical events** (trade policy announcements, geopolitical escalations) with VIX before/after values, which would populate the VIX Spike transform and allow testing of the `d_vix → sp500_ret` edge in the Granger network (the strongest edge, with F=18.6 and p<0.001).

---

*Document generated from codebase state as of February 2026.*
