# Causal Event Study Implementation Plan

## Context

Tremor currently uses Granger causality (predictive association) and directional propagation checks to monitor how shocks flow through financial markets. This is correlation-based — it can't distinguish whether a Fed announcement *caused* a Treasury yield move or whether both responded to the same underlying conditions.

Financial events (Fed announcements, CPI releases, etc.) create **natural experiments** with sharp timing and measurable surprises (`actual - expected`). This makes them ideal for event study methodology — the gold standard for causal identification in financial economics. The goal is to add a module that formally tests: "Does a 1-unit surprise in X cause a statistically significant move in Y?"

## Approach

**OLS dose-response regression** across historical events of a given type. For each event, measure the market response in a narrow daily window, then regress responses on surprise magnitudes. Supplement with placebo/falsification tests and confounding controls. Uses `statsmodels` (already a dependency, currently unused).

## Files to Modify

| File | Change |
|------|--------|
| `tremor/market_data/fetcher.py` | Add `fetch_daily_node_data()` — daily prices without weekly resampling |
| `tremor/config.py` | Add causal test settings |
| `tremor/models/database.py` | Add `CausalTestResult` model |
| `tremor/models/schemas.py` | Add request/response schemas |
| `tremor/app.py` | Register new router |

## Files to Create

| File | Purpose |
|------|---------|
| `tremor/causal/event_study.py` | Core computation: window measurement, OLS regression, placebos, confounding detection |
| `tremor/api/causal_tests.py` | REST endpoints for running and querying tests |
| `tests/test_causal_tests.py` | Tests with synthetic market data |

## Build Order

### Step 1: `tremor/market_data/fetcher.py` — Add daily data function

Add `fetch_daily_node_data(node_name, start_date, end_date) -> pd.Series` that returns raw daily closing prices (no weekly resampling). The existing `_fetch_yahoo` and `_fetch_fred_via_yahoo` already return daily data — this just skips the `resample("W")` step.

### Step 2: `tremor/config.py` — New settings

```python
MIN_EVENTS_FOR_CAUSAL_TEST: int = 5
DEFAULT_PRE_WINDOW_DAYS: int = 5
DEFAULT_POST_WINDOW_DAYS: int = 5
DEFAULT_OVERLAP_BUFFER_DAYS: int = 10
CAUSAL_SIGNIFICANCE_LEVEL: float = 0.05
```

### Step 3: `tremor/models/database.py` — CausalTestResult model

New SQLAlchemy model storing:
- `transform_id` (FK) + `target_node` — what was tested
- Window config: `pre_window_days`, `post_window_days`, `gap_days`
- Sample info: `num_events`, `num_events_used`, `num_events_excluded`, `excluded_event_ids` (JSON)
- Regression output: `coefficient`, `std_error`, `t_statistic`, `p_value`, `r_squared`, `conf_interval_lower`, `conf_interval_upper`, `intercept`, `intercept_p_value`
- Placebo results: `placebo_pre_drift_coeff/pvalue`, `placebo_zero_surprise_coeff/pvalue`
- Assessment: `is_causal` (bool), `confidence_level` ("high"/"medium"/"low"/"none")
- `event_details` (JSON) — per-event surprise + response data for inspection

### Step 4: `tremor/models/schemas.py` — New schemas

- `CausalTestRequest` — input: transform_id, target_node, window params, overlap settings
- `CausalTestResponse` — full output with nested regression/placebo/event detail objects
- `CausalTestSummary` — lightweight for list endpoints
- `RegressionResults`, `PlaceboResults`, `EventStudyDetail` — nested models

### Step 5: `tremor/causal/event_study.py` — Core logic

Main function: `run_event_study(transform_id, target_node, window_params, db) -> CausalTestResult`

#### 5a. Gather data
- Query all Signals for the transform, joined with Events -> extract `(event_id, timestamp, surprise_value)`
- Validate sample size >= `MIN_EVENTS_FOR_CAUSAL_TEST`

#### 5b. Detect confounders
- Query ALL events in the system within the study date range
- For each study event, check if any other event falls within `overlap_buffer_days`
- Build exclusion list with reasons

#### 5c. Fetch market data
- One call to `fetch_daily_node_data` covering the full date range (earliest event - pre_window to latest event + post_window)

#### 5d. Compute window returns per event
- For each event timestamp, find nearest trading days for window boundaries
- `pre_return = log(price[t - gap]) - log(price[t - pre_window])`
- `post_return = log(price[t + post_window]) - log(price[t + gap])`

#### 5e. Dose-response regression
```python
import statsmodels.api as sm
X = sm.add_constant(surprise_values)
model = sm.OLS(post_returns, X).fit(cov_type='HC1')  # White heteroskedasticity-robust SEs
```
Extract: coefficient, std_error, t_stat, p_value, R-squared, confidence interval

#### 5f. Placebo test 1 — Pre-event drift
- Regress pre-window returns on surprise magnitudes
- Should be insignificant (no information leakage before event)

#### 5g. Placebo test 2 — Zero-surprise events
- Filter events where `abs(surprise) < 0.5 * std(all surprises)`
- Test if their mean response differs from zero (intercept-only OLS)
- Should be insignificant (non-events produce no response)

#### 5h. Confidence assessment
- **"high"**: p < 0.01, both placebos pass, R-squared > 0.15, N >= 10
- **"medium"**: p < 0.05, at least one placebo passes, N >= 7
- **"low"**: p < 0.10, N >= 5
- **"none"**: p >= 0.10 or N < 5
- `is_causal` = True if "high" or "medium"

### Step 6: `tremor/api/causal_tests.py` — Endpoints

```
POST   /causal-tests/run                    Run event study, persist + return results
GET    /causal-tests                         List past results (filter by transform, target, is_causal)
GET    /causal-tests/feasibility             Check which transform-target pairs have enough events
GET    /causal-tests/{test_id}               Full result detail
DELETE /causal-tests/{test_id}               Delete result
```

### Step 7: `tremor/app.py` — Register router

```python
from tremor.api import causal_tests
app.include_router(causal_tests.router)
```

### Step 8: `tests/test_causal_tests.py` — Tests

Mock `fetch_daily_node_data` with synthetic price series. Key tests:

1. **Basic test** — 8+ events with linear surprise-response relationship -> significant coefficient
2. **Insufficient events** — <5 events -> 400 error
3. **No effect** — random noise market data -> `is_causal: False`
4. **Confounding exclusion** — overlapping events excluded when flag is on
5. **Placebo pre-drift** — pre-correlated data -> `pre_drift_passed: False`
6. **Placebo zero-surprise** — non-surprise events show no response -> `zero_surprise_passed: True`
7. **List/get endpoints** — persisted results retrievable
8. **Feasibility endpoint** — correct reporting of testable pairs

## Statistical Methodology

### Why OLS with HC1 standard errors?
OLS dose-response regression is the standard event study methodology in financial economics (MacKinlay 1997). The coefficient directly answers "how much does a 1-unit surprise move the target variable?" HC1 (White) standard errors handle heteroskedasticity without imposing structure. With the small samples typical here (5-30 events), more complex estimators (GMM, bootstrap) would be unreliable.

### Why daily granularity?
Weekly data conflates the event response with noise from the entire week. Daily windows of 1-5 days around the event are standard in the literature and dramatically improve statistical power.

### Why all events, not just shocks?
The dose-response approach uses the full range of surprise magnitudes, not just extremes. This provides more statistical power. Zero-surprise events serve as a natural placebo control group — they should show no market response if the causal claim is valid.

### Why exclusion over multivariate control for confounders?
With small samples (5-30 events), adding control variables burns degrees of freedom. Exclusion is more conservative and easier to interpret. The user can override this via `exclude_overlapping=False`.

### Why log returns?
Log returns are additive across time, approximately normally distributed, and symmetric. This matters for the OLS normality assumptions and makes window return computation cleaner.

## Verification

1. Run `pytest tests/test_causal_tests.py` — all 8 test cases pass
2. Run `pytest tests/` — existing tests still pass (no regressions)
3. Start server (`uvicorn tremor.app:app`), seed transforms, create 8+ Fed events with varying surprises, call `POST /causal-tests/run` with a transform_id and target_node, verify response structure and statistical output
4. Check `GET /causal-tests/feasibility` reports correct testable pairs
