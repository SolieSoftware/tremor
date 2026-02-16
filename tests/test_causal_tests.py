from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd


def _create_transform(client, name="Fed Rate Surprise"):
    resp = client.post("/signals/transforms", json={
        "name": name,
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "unit": "percent",
    })
    return resp.json()


def _create_event_and_compute(client, timestamp, expected, actual):
    resp = client.post("/events", json={
        "timestamp": timestamp,
        "type": "fed_announcement",
        "description": "FOMC decision",
        "raw_data": {"expected_rate": expected, "actual_rate": actual},
    })
    event_id = resp.json()["id"]
    client.post(f"/signals/compute/{event_id}")
    return event_id


def _make_synthetic_prices(event_dates, surprises, effect_size, noise_sd=0.001):
    """Generate daily price series where post-event returns correlate with surprise.

    Creates a date range covering all events, with prices that respond to
    surprises in the post-event window.
    """
    earliest = min(event_dates) - timedelta(days=30)
    latest = max(event_dates) + timedelta(days=30)
    dates = pd.bdate_range(earliest, latest)

    rng = np.random.RandomState(42)
    prices = np.ones(len(dates)) * 100.0

    # Build cumulative random walk
    for i in range(1, len(dates)):
        prices[i] = prices[i - 1] * (1 + rng.normal(0, noise_sd))

    # Inject effect: for each event, bump prices after the event proportional to surprise
    for event_date, surprise in zip(event_dates, surprises):
        event_ts = pd.Timestamp(event_date)
        # Find the index closest to the event date
        mask = dates >= event_ts
        if not mask.any():
            continue
        start_idx = np.argmax(mask)
        # Apply effect over post-window (5 days after event)
        for j in range(start_idx, min(start_idx + 5, len(dates))):
            prices[j] *= (1 + effect_size * surprise / 5)

    return pd.Series(prices, index=dates)


def _seed_events_with_surprises(client, num_events=8, base_rate=4.0):
    """Create a set of events with varying surprises spread over time."""
    rng = np.random.RandomState(123)
    surprises = rng.uniform(-0.5, 0.5, num_events)
    event_dates = []
    event_ids = []

    base_date = datetime(2024, 1, 15)
    for i in range(num_events):
        ts = base_date + timedelta(days=30 * i)  # ~monthly spacing
        event_dates.append(ts)
        surprise = surprises[i]
        event_id = _create_event_and_compute(
            client,
            timestamp=ts.isoformat() + "Z",
            expected=base_rate,
            actual=base_rate + surprise,
        )
        event_ids.append(event_id)

    return event_dates, surprises, event_ids


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_run_causal_test_basic(mock_fetch, client):
    """Test that a causal test runs and returns significant results with correlated data."""
    transform = _create_transform(client)
    event_dates, surprises, _ = _seed_events_with_surprises(client)

    prices = _make_synthetic_prices(event_dates, surprises, effect_size=0.05)
    mock_fetch.return_value = prices

    resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "pre_window_days": 5,
        "post_window_days": 5,
        "exclude_overlapping": False,
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["num_events"] == 8
    assert data["num_events_used"] >= 5
    assert "regression" in data
    assert "placebo" in data
    assert data["regression"]["num_observations"] >= 5
    assert data["regression"]["coefficient"] != 0
    assert data["id"] is not None


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_run_causal_test_insufficient_events(mock_fetch, client):
    """Test that too few events returns 400."""
    transform = _create_transform(client)

    # Create only 3 events (minimum is 5)
    base_date = datetime(2024, 1, 15)
    for i in range(3):
        ts = base_date + timedelta(days=30 * i)
        _create_event_and_compute(
            client,
            timestamp=ts.isoformat() + "Z",
            expected=4.0,
            actual=4.25,
        )

    resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
    })
    assert resp.status_code == 400
    assert "Insufficient" in resp.json()["detail"]


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_causal_test_no_effect(mock_fetch, client):
    """Test that random noise data yields is_causal=False."""
    transform = _create_transform(client)
    event_dates, _, _ = _seed_events_with_surprises(client)

    # Pure noise prices — no relationship to surprises
    prices = _make_synthetic_prices(event_dates, [0] * len(event_dates), effect_size=0)
    mock_fetch.return_value = prices

    resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "exclude_overlapping": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_causal"] is False


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_confounding_exclusion(mock_fetch, client):
    """Test that overlapping events are excluded when flag is on."""
    transform = _create_transform(client)

    # Create 8 events, two of which are only 5 days apart (within default 10-day buffer)
    # After excluding the overlapping pair, 6 remain (above minimum of 5)
    base_date = datetime(2024, 1, 15)
    dates_and_surprises = [
        (base_date, 0.25),
        (base_date + timedelta(days=5), 0.30),  # overlaps with first
        (base_date + timedelta(days=60), -0.10),
        (base_date + timedelta(days=90), 0.50),
        (base_date + timedelta(days=120), -0.25),
        (base_date + timedelta(days=150), 0.15),
        (base_date + timedelta(days=180), -0.35),
        (base_date + timedelta(days=210), 0.40),
    ]

    event_dates = []
    for ts, surprise in dates_and_surprises:
        event_dates.append(ts)
        _create_event_and_compute(
            client,
            timestamp=ts.isoformat() + "Z",
            expected=4.0,
            actual=4.0 + surprise,
        )

    surprises = [s for _, s in dates_and_surprises]
    prices = _make_synthetic_prices(event_dates, surprises, effect_size=0.05)
    mock_fetch.return_value = prices

    # With exclusion
    resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "exclude_overlapping": True,
        "overlap_buffer_days": 10,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_events_excluded"] > 0
    assert data["num_events_used"] < data["num_events"]


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_placebo_zero_surprise(mock_fetch, client):
    """Test that zero-surprise events produce a placebo result."""
    transform = _create_transform(client)

    base_date = datetime(2024, 1, 15)
    surprises_list = [0.0, 0.01, -0.01, 0.5, -0.5, 0.3, -0.3, 0.02]
    event_dates = []
    for i, surprise in enumerate(surprises_list):
        ts = base_date + timedelta(days=30 * i)
        event_dates.append(ts)
        _create_event_and_compute(
            client,
            timestamp=ts.isoformat() + "Z",
            expected=4.0,
            actual=4.0 + surprise,
        )

    prices = _make_synthetic_prices(event_dates, surprises_list, effect_size=0.05)
    mock_fetch.return_value = prices

    resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "exclude_overlapping": False,
    })
    assert resp.status_code == 200
    data = resp.json()
    # Should have placebo results since we have near-zero surprise events
    assert data["placebo"]["zero_surprise_p_value"] is not None


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_list_causal_tests(mock_fetch, client):
    """Test that completed tests are retrievable via list endpoint."""
    transform = _create_transform(client)
    event_dates, surprises, _ = _seed_events_with_surprises(client)

    prices = _make_synthetic_prices(event_dates, surprises, effect_size=0.05)
    mock_fetch.return_value = prices

    client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "exclude_overlapping": False,
    })

    resp = client.get("/causal-tests")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["transform_id"] == transform["id"]


@patch("tremor.causal.event_study.fetch_daily_node_data")
def test_get_causal_test_detail(mock_fetch, client):
    """Test retrieving a specific test by ID."""
    transform = _create_transform(client)
    event_dates, surprises, _ = _seed_events_with_surprises(client)

    prices = _make_synthetic_prices(event_dates, surprises, effect_size=0.05)
    mock_fetch.return_value = prices

    run_resp = client.post("/causal-tests/run", json={
        "transform_id": transform["id"],
        "target_node": "d_treasury_10y",
        "exclude_overlapping": False,
    })
    test_id = run_resp.json()["id"]

    resp = client.get(f"/causal-tests/{test_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == test_id
    assert len(data["event_details"]) > 0


def test_feasibility_endpoint(client):
    """Test that feasibility correctly reports testable pairs."""
    transform = _create_transform(client)

    # No events yet
    resp = client.get("/causal-tests/feasibility")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["feasible"] is False
    assert results[0]["num_events"] == 0

    # Add 5 events with signals
    base_date = datetime(2024, 1, 15)
    for i in range(5):
        ts = base_date + timedelta(days=30 * i)
        _create_event_and_compute(
            client,
            timestamp=ts.isoformat() + "Z",
            expected=4.0,
            actual=4.25,
        )

    resp = client.get("/causal-tests/feasibility")
    results = resp.json()
    assert results[0]["feasible"] is True
    assert results[0]["num_events"] == 5
