from unittest.mock import patch

import networkx as nx


def _setup_network():
    """Set up a test causal network."""
    from tremor.causal.network import causal_network

    causal_network.clear()
    causal_network.add_edge("d_fed_funds", "d_treasury_10y", f_statistic=15.0, p_value=0.001, lag=1)
    causal_network.add_edge("d_fed_funds", "d_vix", f_statistic=8.0, p_value=0.01, lag=2)
    causal_network.add_edge("d_treasury_10y", "d_credit_spread", f_statistic=10.0, p_value=0.005, lag=1)


def test_shock_creates_propagation_monitors(client):
    _setup_network()

    # Register transform
    client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "threshold_sd": 0.1,
    })

    # Create event with a large surprise (shock)
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Surprise rate hike",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 5.875},
    })
    event_id = resp.json()["id"]

    # Compute signals — should detect a shock
    resp = client.post(f"/signals/compute/{event_id}")
    signals = resp.json()
    assert len(signals) == 1
    assert signals[0]["is_shock"] is True

    signal_id = signals[0]["id"]

    # Get propagation — should create monitoring records for downstream nodes
    resp = client.get(f"/monitor/shocks/{signal_id}/propagation")
    assert resp.status_code == 200
    props = resp.json()
    assert len(props) == 2  # d_treasury_10y and d_vix

    target_nodes = {p["target_node"] for p in props}
    assert target_nodes == {"d_treasury_10y", "d_vix"}

    for p in props:
        assert p["source_node"] == "d_fed_funds"
        assert p["status"] == "monitoring"


def test_propagation_check(client):
    _setup_network()

    client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "threshold_sd": 0.1,
    })

    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Surprise rate hike",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 5.875},
    })
    event_id = resp.json()["id"]
    resp = client.post(f"/signals/compute/{event_id}")
    signal_id = resp.json()[0]["id"]

    # Create propagation monitors
    client.get(f"/monitor/shocks/{signal_id}/propagation")

    # Mock market data fetch to avoid real API calls
    import pandas as pd

    mock_data = pd.Series([0.0, 0.1, 0.15], index=pd.date_range("2024-12-18", periods=3, freq="W"))

    with patch("tremor.core.propagation.fetch_node_data", return_value=mock_data):
        resp = client.post(f"/monitor/shocks/{signal_id}/check")
        assert resp.status_code == 200
        props = resp.json()
        assert len(props) == 2

        for p in props:
            assert p["actual_change"] is not None


def test_network_endpoint(client):
    _setup_network()

    resp = client.get("/monitor/network")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_nodes"] == 4  # d_fed_funds, d_treasury_10y, d_vix, d_credit_spread
    assert data["total_edges"] == 3
    assert len(data["edges"]) == 3
