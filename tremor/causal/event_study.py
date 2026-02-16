"""Causal event study module.

Implements OLS dose-response regression to test whether event surprises
cause statistically significant market responses. Includes placebo tests
for pre-event drift and zero-surprise controls.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import statsmodels.api as sm
from sqlalchemy.orm import Session

from tremor.config import settings
from tremor.market_data.fetcher import fetch_daily_node_data
from tremor.models.database import CausalTestResult, Event, Signal, SignalTransform


def run_event_study(
    transform_id: str,
    target_node: str,
    pre_window_days: int,
    post_window_days: int,
    gap_days: int,
    exclude_overlapping: bool,
    overlap_buffer_days: int,
    significance_level: float,
    db: Session,
) -> CausalTestResult:
    """Run a causal event study for a transform-target pair.

    Gathers all historical events that produced signals for the given transform,
    fetches daily market data around each event, runs OLS regression of market
    responses on surprise magnitudes, and executes placebo tests.
    """
    # 1. Gather signals with their events
    signals_with_events = (
        db.query(Signal, Event)
        .join(Event, Signal.event_id == Event.id)
        .filter(Signal.transform_id == transform_id)
        .order_by(Event.timestamp)
        .all()
    )

    num_events = len(signals_with_events)
    if num_events < settings.MIN_EVENTS_FOR_CAUSAL_TEST:
        raise ValueError(
            f"Insufficient events: {num_events} found, "
            f"{settings.MIN_EVENTS_FOR_CAUSAL_TEST} required"
        )

    # Extract study data
    study_events = [
        {
            "event_id": event.id,
            "timestamp": _ensure_aware(event.timestamp),
            "surprise": signal.value,
        }
        for signal, event in signals_with_events
    ]

    # 2. Detect confounding (overlapping events)
    exclusions = {}
    if exclude_overlapping:
        exclusions = _detect_overlapping_events(
            study_events, overlap_buffer_days, db
        )

    # Build event details and split into included/excluded
    event_details = []
    included_events = []
    for ev in study_events:
        excluded = ev["event_id"] in exclusions
        detail = {
            "event_id": ev["event_id"],
            "event_timestamp": ev["timestamp"].isoformat(),
            "surprise_value": ev["surprise"],
            "pre_window_return": None,
            "post_window_return": None,
            "excluded": excluded,
            "exclusion_reason": exclusions.get(ev["event_id"]),
        }
        event_details.append(detail)
        if not excluded:
            included_events.append(ev)

    num_events_used = len(included_events)
    if num_events_used < settings.MIN_EVENTS_FOR_CAUSAL_TEST:
        raise ValueError(
            f"Insufficient events after exclusions: {num_events_used} remaining, "
            f"{settings.MIN_EVENTS_FOR_CAUSAL_TEST} required"
        )

    # 3. Fetch daily market data for the full date range
    earliest = min(ev["timestamp"] for ev in included_events)
    latest = max(ev["timestamp"] for ev in included_events)
    fetch_start = earliest - timedelta(days=pre_window_days + gap_days + 10)  # buffer for weekends
    fetch_end = latest + timedelta(days=post_window_days + gap_days + 10)

    prices = fetch_daily_node_data(target_node, fetch_start, fetch_end)
    if prices is None or prices.empty:
        raise ValueError(f"No market data available for node '{target_node}'")

    # 4. Compute window returns for each included event
    surprises = []
    pre_returns = []
    post_returns = []

    for i, ev in enumerate(included_events):
        pre_ret, post_ret = _compute_window_returns(
            ev["timestamp"], prices, pre_window_days, post_window_days, gap_days
        )
        if pre_ret is None or post_ret is None:
            # Mark as excluded due to missing data
            for detail in event_details:
                if detail["event_id"] == ev["event_id"]:
                    detail["excluded"] = True
                    detail["exclusion_reason"] = "insufficient market data in window"
            continue

        surprises.append(ev["surprise"])
        pre_returns.append(pre_ret)
        post_returns.append(post_ret)

        # Update event details with computed returns
        for detail in event_details:
            if detail["event_id"] == ev["event_id"]:
                detail["pre_window_return"] = pre_ret
                detail["post_window_return"] = post_ret

    num_events_used = len(surprises)
    if num_events_used < settings.MIN_EVENTS_FOR_CAUSAL_TEST:
        raise ValueError(
            f"Insufficient events with valid market data: {num_events_used} remaining, "
            f"{settings.MIN_EVENTS_FOR_CAUSAL_TEST} required"
        )

    surprises_arr = np.array(surprises)
    pre_returns_arr = np.array(pre_returns)
    post_returns_arr = np.array(post_returns)

    # 5. Dose-response regression: post_return ~ surprise
    reg = _run_ols_regression(surprises_arr, post_returns_arr, significance_level)

    # 6. Placebo test 1: pre-event drift
    pre_drift = _run_placebo_pre_drift(surprises_arr, pre_returns_arr, significance_level)

    # 7. Placebo test 2: zero-surprise events
    zero_surprise = _run_placebo_zero_surprise(
        surprises_arr, post_returns_arr, significance_level
    )

    # 8. Confidence assessment
    is_causal, confidence = _assess_confidence(
        reg, pre_drift, zero_surprise, num_events_used
    )

    # 9. Build and return result
    excluded_ids = [d["event_id"] for d in event_details if d["excluded"]]

    result = CausalTestResult(
        transform_id=transform_id,
        target_node=target_node,
        pre_window_days=pre_window_days,
        post_window_days=post_window_days,
        gap_days=gap_days,
        num_events=num_events,
        num_events_used=num_events_used,
        num_events_excluded=len(excluded_ids),
        excluded_event_ids=excluded_ids,
        coefficient=reg["coefficient"],
        std_error=reg["std_error"],
        t_statistic=reg["t_statistic"],
        p_value=reg["p_value"],
        r_squared=reg["r_squared"],
        conf_interval_lower=reg["conf_interval_lower"],
        conf_interval_upper=reg["conf_interval_upper"],
        intercept=reg["intercept"],
        intercept_p_value=reg["intercept_p_value"],
        placebo_pre_drift_coeff=pre_drift.get("coefficient"),
        placebo_pre_drift_pvalue=pre_drift.get("p_value"),
        placebo_zero_surprise_coeff=zero_surprise.get("coefficient"),
        placebo_zero_surprise_pvalue=zero_surprise.get("p_value"),
        is_causal=is_causal,
        confidence_level=confidence,
        event_details=event_details,
    )

    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _detect_overlapping_events(
    study_events: list[dict],
    buffer_days: int,
    db: Session,
) -> dict[str, str]:
    """Find study events that have other events within buffer_days.

    Returns {event_id: exclusion_reason} for confounded events.
    """
    study_ids = {ev["event_id"] for ev in study_events}
    earliest = min(ev["timestamp"] for ev in study_events) - timedelta(days=buffer_days)
    latest = max(ev["timestamp"] for ev in study_events) + timedelta(days=buffer_days)

    all_events = (
        db.query(Event)
        .filter(Event.timestamp >= earliest, Event.timestamp <= latest)
        .all()
    )

    exclusions: dict[str, str] = {}
    for study_ev in study_events:
        for other in all_events:
            if other.id in study_ids and other.id == study_ev["event_id"]:
                continue
            other_ts = _ensure_aware(other.timestamp)
            delta = abs((study_ev["timestamp"] - other_ts).total_seconds()) / 86400
            if delta <= buffer_days:
                exclusions[study_ev["event_id"]] = (
                    f"overlapping with event '{other.id}' "
                    f"({other.type}, {delta:.1f} days apart)"
                )
                break

    return exclusions


def _compute_window_returns(
    event_ts: datetime,
    prices: "pd.Series",
    pre_window_days: int,
    post_window_days: int,
    gap_days: int,
) -> tuple[Optional[float], Optional[float]]:
    """Compute pre-window and post-window log returns around an event.

    Finds the nearest available trading days for window boundaries.
    Returns (pre_return, post_return) or (None, None) if data is insufficient.
    """
    event_date = event_ts.date()

    # Find nearest trading day indices
    pre_start_date = event_date - timedelta(days=pre_window_days + gap_days)
    pre_end_date = event_date - timedelta(days=gap_days) if gap_days > 0 else event_date
    post_start_date = event_date + timedelta(days=gap_days) if gap_days > 0 else event_date
    post_end_date = event_date + timedelta(days=post_window_days + gap_days)

    pre_start_price = _get_nearest_price(prices, pre_start_date, direction="backward")
    pre_end_price = _get_nearest_price(prices, pre_end_date, direction="backward")
    post_start_price = _get_nearest_price(prices, post_start_date, direction="forward")
    post_end_price = _get_nearest_price(prices, post_end_date, direction="forward")

    if any(p is None or p <= 0 for p in [pre_start_price, pre_end_price, post_start_price, post_end_price]):
        return None, None

    pre_return = float(np.log(pre_end_price / pre_start_price))
    post_return = float(np.log(post_end_price / post_start_price))

    return pre_return, post_return


def _get_nearest_price(
    prices: "pd.Series",
    target_date,
    direction: str = "backward",
    max_search_days: int = 7,
) -> Optional[float]:
    """Find the nearest available price to a target date.

    Searches backward or forward up to max_search_days.
    """
    import pandas as pd

    target = pd.Timestamp(target_date)
    if target.tzinfo is None and prices.index.tz is not None:
        target = target.tz_localize(prices.index.tz)
    elif target.tzinfo is not None and prices.index.tz is None:
        target = target.tz_localize(None)

    for i in range(max_search_days + 1):
        offset = timedelta(days=i)
        check_date = target - offset if direction == "backward" else target + offset
        if check_date in prices.index:
            val = prices[check_date]
            return float(val)

    return None


def _run_ols_regression(
    surprises: np.ndarray,
    responses: np.ndarray,
    significance_level: float,
) -> dict:
    """Run OLS: response ~ surprise with HC1 robust standard errors."""
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
        "intercept": float(results.params[0]),
        "intercept_p_value": float(results.pvalues[0]),
        "num_observations": int(results.nobs),
    }


def _run_placebo_pre_drift(
    surprises: np.ndarray,
    pre_returns: np.ndarray,
    significance_level: float,
) -> dict:
    """Placebo test: regress pre-window returns on surprise magnitudes.

    A significant coefficient suggests information leakage or endogeneity.
    """
    X = sm.add_constant(surprises)
    model = sm.OLS(pre_returns, X)
    results = model.fit(cov_type="HC1")

    return {
        "coefficient": float(results.params[1]),
        "p_value": float(results.pvalues[1]),
        "passed": bool(results.pvalues[1] > significance_level),
    }


def _run_placebo_zero_surprise(
    surprises: np.ndarray,
    responses: np.ndarray,
    significance_level: float,
) -> dict:
    """Placebo test: check that near-zero surprises produce no market response.

    Filters events with abs(surprise) < 0.5 * std(surprises) and tests
    whether their mean response differs from zero.
    """
    surprise_std = np.std(surprises)
    if surprise_std == 0:
        return {"coefficient": None, "p_value": None, "passed": None}

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


def _assess_confidence(
    reg: dict,
    pre_drift: dict,
    zero_surprise: dict,
    num_events: int,
) -> tuple[Optional[bool], Optional[str]]:
    """Assess overall causal confidence based on regression and placebo results."""
    p = reg["p_value"]
    r2 = reg["r_squared"]

    pre_passed = pre_drift.get("passed")
    zero_passed = zero_surprise.get("passed")

    placebos_passed = sum(1 for x in [pre_passed, zero_passed] if x is True)
    placebos_available = sum(1 for x in [pre_passed, zero_passed] if x is not None)

    if p < 0.01 and r2 > 0.15 and num_events >= 10 and placebos_passed == placebos_available and placebos_available > 0:
        return True, "high"
    elif p < 0.05 and num_events >= 7 and placebos_passed >= 1:
        return True, "medium"
    elif p < 0.10 and num_events >= 5:
        return False, "low"
    else:
        return False, "none"
