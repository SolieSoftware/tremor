def test_create_event(client):
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "subtype": "rate_decision",
        "description": "FOMC rate decision",
        "tags": ["fomc"],
        "raw_data": {"expected_rate": 4.375, "actual_rate": 4.375},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "fed_announcement"
    assert data["raw_data"]["expected_rate"] == 4.375


def test_list_events(client):
    client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Event 1",
    })
    client.post("/events", json={
        "timestamp": "2024-12-19T14:00:00Z",
        "type": "earnings",
        "description": "Event 2",
    })
    resp = client.get("/events")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_list_events_filter_by_type(client):
    client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Fed event",
    })
    client.post("/events", json={
        "timestamp": "2024-12-19T14:00:00Z",
        "type": "earnings",
        "description": "Earnings event",
    })
    resp = client.get("/events", params={"type": "fed_announcement"})
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 1
    assert events[0]["type"] == "fed_announcement"


def test_get_event(client):
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "Test event",
    })
    event_id = resp.json()["id"]
    resp = client.get(f"/events/{event_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == event_id


def test_get_event_not_found(client):
    resp = client.get("/events/nonexistent")
    assert resp.status_code == 404


def test_delete_event(client):
    resp = client.post("/events", json={
        "timestamp": "2024-12-18T14:00:00Z",
        "type": "fed_announcement",
        "description": "To delete",
    })
    event_id = resp.json()["id"]
    resp = client.delete(f"/events/{event_id}")
    assert resp.status_code == 200

    resp = client.get(f"/events/{event_id}")
    assert resp.status_code == 404
