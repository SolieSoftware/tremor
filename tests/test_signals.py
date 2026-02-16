def test_create_transform(client):
    resp = client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "unit": "percent",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Fed Rate Surprise"
    assert data["node_mapping"] == "d_fed_funds"


def test_list_transforms(client):
    client.post("/signals/transforms", json={
        "name": "Transform 1",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
    })
    client.post("/signals/transforms", json={
        "name": "Transform 2",
        "event_types": ["earnings"],
        "transform_expression": "actual_eps - expected_eps",
        "node_mapping": "sp500_ret",
    })
    resp = client.get("/signals/transforms")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_transform(client):
    resp = client.post("/signals/transforms", json={
        "name": "To Delete",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
    })
    tid = resp.json()["id"]
    resp = client.delete(f"/signals/transforms/{tid}")
    assert resp.status_code == 200

    resp = client.get(f"/signals/transforms/{tid}")
    assert resp.status_code == 404


def test_compute_signals(client):
    # Register a transform
    client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "unit": "percent",
    })

    # Create an event
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "FOMC rate decision",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 4.625},
    })
    event_id = resp.json()["id"]

    # Compute signals
    resp = client.post(f"/signals/compute/{event_id}")
    assert resp.status_code == 200
    signals = resp.json()
    assert len(signals) == 1
    assert signals[0]["value"] == 0.25  # 4.625 - 4.375


def test_compute_signals_shock_detection(client):
    # Register a transform with low threshold
    client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "threshold_sd": 2.0,
    })

    # With fewer than 5 historical signals, absolute threshold is used
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Big surprise",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 5.875},
    })
    event_id = resp.json()["id"]
    resp = client.post(f"/signals/compute/{event_id}")
    signals = resp.json()
    assert len(signals) == 1
    assert signals[0]["is_shock"] is True  # 1.5 > 1.0 absolute threshold


def test_list_signals(client):
    client.post("/signals/transforms", json={
        "name": "Fed Rate Surprise",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
    })
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Test",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 4.625},
    })
    event_id = resp.json()["id"]
    client.post(f"/signals/compute/{event_id}")

    resp = client.get("/signals")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_no_matching_transform(client):
    # Create event with no matching transforms
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "No transforms registered",
        "raw_data": {"expected_rate": 4.375, "actual_rate": 4.625},
    })
    event_id = resp.json()["id"]
    resp = client.post(f"/signals/compute/{event_id}")
    assert resp.status_code == 200
    assert len(resp.json()) == 0
