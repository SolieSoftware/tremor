"""Full pipeline smoke test for Tremor.

Walks through the complete flow:
  1. Seed transforms
  2. Register an event
  3. Compute signals (shock detection)
  4. Inspect shock status
  5. Trigger propagation monitoring (if network loaded)
  6. Print DB summary

Usage:
    # Start the server first: uvicorn tremor.app:app --reload
    # Then in another terminal:
    python scripts/test_pipeline.py
"""

import json
import sys
from datetime import datetime, timezone

import httpx

BASE = "http://localhost:8000"


def pp(label, resp):
    print(f"\n--- {label} [{resp.status_code}] ---")
    try:
        data = resp.json()
        print(json.dumps(data, indent=2)[:800])
    except Exception:
        print(resp.text[:400])
    if resp.status_code >= 400:
        print("  [ERROR] Stopping.")
        sys.exit(1)
    return resp.json()


def run():
    client = httpx.Client(base_url=BASE, timeout=30)

    # ── 1. Seed transforms ──────────────────────────────────────────────────
    print("\n========================================")
    print("STEP 1: Seed signal transforms")
    print("========================================")
    resp = client.get("/signals/transforms")
    transforms = resp.json()
    if not transforms:
        print("  No transforms found — seed via: python scripts/seed_transforms.py")
        print("  Continuing to register one manually...")
        body = {
            "name": "Fed Rate Surprise",
            "description": "Rate surprise test",
            "event_types": ["fed_announcement"],
            "transform_expression": "actual_rate - expected_rate",
            "node_mapping": "d_fed_funds",
            "unit": "percent",
            "threshold_sd": 2.0,
        }
        resp = client.post("/signals/transforms", json=body)
        transforms = [pp("Register transform", resp)]
    else:
        print(f"  {len(transforms)} transforms already registered.")
        for t in transforms[:3]:
            print(f"    [{t['id'][:8]}] {t['name']}  →  {t['node_mapping']}")

    # Find the Fed Rate Surprise transform
    fed_transform = next((t for t in transforms if "Fed Rate" in t["name"]), transforms[0])
    print(f"\n  Using transform: {fed_transform['name']} (id={fed_transform['id'][:8]})")

    # ── 2. Register a Fed announcement event ───────────────────────────────
    print("\n========================================")
    print("STEP 2: Register a Fed announcement event")
    print("========================================")
    event_body = {
        "timestamp": "2024-11-07T14:00:00Z",
        "type": "fed_announcement",
        "subtype": "rate_decision",
        "description": "FOMC Nov 2024 — 25bp cut, slightly hawkish statement",
        "tags": ["fomc", "rate_cut", "2024"],
        "raw_data": {
            "expected_rate": 4.875,
            "actual_rate": 4.625,
            "change_bps": -25,
            "statement_tone": "hawkish",
        },
    }
    resp = client.post("/events", json=event_body)
    event = pp("Create event", resp)
    event_id = event["id"]

    # ── 3. Register a second event for z-score context ─────────────────────
    print("\n========================================")
    print("STEP 3: Register additional events (for z-score baseline)")
    print("========================================")
    historical_events = [
        {
            "timestamp": "2024-09-18T14:00:00Z",
            "type": "fed_announcement",
            "subtype": "rate_decision",
            "description": "FOMC Sep 2024 — 50bp cut",
            "tags": ["fomc", "rate_cut"],
            "raw_data": {"expected_rate": 5.375, "actual_rate": 4.875, "change_bps": -50},
        },
        {
            "timestamp": "2024-07-31T14:00:00Z",
            "type": "fed_announcement",
            "subtype": "rate_decision",
            "description": "FOMC Jul 2024 — hold, as expected",
            "tags": ["fomc", "hold"],
            "raw_data": {"expected_rate": 5.375, "actual_rate": 5.375, "change_bps": 0},
        },
        {
            "timestamp": "2024-05-01T14:00:00Z",
            "type": "fed_announcement",
            "subtype": "rate_decision",
            "description": "FOMC May 2024 — hold",
            "tags": ["fomc", "hold"],
            "raw_data": {"expected_rate": 5.375, "actual_rate": 5.375, "change_bps": 0},
        },
        {
            "timestamp": "2024-03-20T14:00:00Z",
            "type": "fed_announcement",
            "subtype": "rate_decision",
            "description": "FOMC Mar 2024 — hold",
            "tags": ["fomc", "hold"],
            "raw_data": {"expected_rate": 5.375, "actual_rate": 5.375, "change_bps": 0},
        },
    ]
    historical_ids = []
    for ev in historical_events:
        r = client.post("/events", json=ev)
        if r.status_code == 200:
            hid = r.json()["id"]
            historical_ids.append(hid)
            print(f"  Registered: {ev['description'][:50]}")
            # Compute signals for historical events so z-score has a baseline
            client.post(f"/signals/compute/{hid}")

    # ── 4. Compute signals for our main event ──────────────────────────────
    print("\n========================================")
    print("STEP 4: Compute signals for main event")
    print("========================================")
    resp = client.post(f"/signals/compute/{event_id}")
    signals = pp("Compute signals", resp)
    if signals:
        for s in signals:
            shock = " *** SHOCK DETECTED ***" if s["is_shock"] else ""
            print(f"\n  Signal: {s['value']:.4f}  z={s.get('z_score', 'N/A')}  {shock}")

    # ── 5. List events ─────────────────────────────────────────────────────
    print("\n========================================")
    print("STEP 5: List events")
    print("========================================")
    resp = client.get("/events", params={"type": "fed_announcement", "limit": 10})
    events = pp("List fed_announcement events", resp)
    print(f"\n  Total: {len(events)} events")

    # ── 6. Check shocks ────────────────────────────────────────────────────
    print("\n========================================")
    print("STEP 6: Check for shocks")
    print("========================================")
    resp = client.get("/monitor/shocks")
    shocks = pp("List shocks", resp)
    print(f"\n  Total shocks detected: {len(shocks)}")
    for s in shocks[:3]:
        print(f"    signal_id={s['id'][:8]}  value={s['value']:.4f}  z={s.get('z_score')}")

    # ── 7. Propagation for shocks ──────────────────────────────────────────
    if shocks:
        shock_id = shocks[0]["id"]
        print("\n========================================")
        print(f"STEP 7: View propagation for shock {shock_id[:8]}")
        print("========================================")
        resp = client.get(f"/monitor/shocks/{shock_id}/propagation")
        pp("Propagation results", resp)

    # ── 8. Network status ──────────────────────────────────────────────────
    print("\n========================================")
    print("STEP 8: Causal network status")
    print("========================================")
    resp = client.get("/monitor/network")
    network = pp("Network", resp)

    # ── 9. DB summary ──────────────────────────────────────────────────────
    print("\n========================================")
    print("DONE — run database CLI for more detail:")
    print("  python scripts/db_cli.py status")
    print("  python scripts/db_cli.py events")
    print("  python scripts/db_cli.py signals --shock")
    print("  python scripts/db_cli.py shell")
    print("========================================\n")


if __name__ == "__main__":
    try:
        resp = httpx.get(f"{BASE}/docs", timeout=3)
    except Exception:
        print(f"ERROR: Cannot reach {BASE}")
        print("Start the server first: uvicorn tremor.app:app --reload")
        sys.exit(1)
    run()
