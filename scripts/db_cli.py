"""Interactive CLI for browsing and querying the Tremor SQLite database.

Usage:
    python scripts/db_cli.py [command] [options]

Commands:
    status          — summary counts for all tables
    events          — list events (--type, --limit)
    transforms      — list signal transforms
    signals         — list computed signals (--shock, --limit)
    propagation     — list propagation results (--status, --limit)
    causal          — list causal test results (--limit)
    event <id>      — show full event detail including signals
    shell           — drop into an interactive SQLite shell
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Make tremor importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tremor.models.database import (
    CausalTestResult,
    Event,
    PropagationResult,
    Signal,
    SignalTransform,
    SessionLocal,
    init_db,
)
from tremor.config import settings


def fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "—"


def fmt_float(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "—"


def print_table(rows, headers):
    if not rows:
        print("  (no rows)")
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  " + "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def cmd_status(db, args):
    print("\n=== Tremor Database Status ===")
    print(f"  Database: {settings.DATABASE_URL}")
    print()
    counts = [
        ("events", db.query(Event).count()),
        ("signal_transforms", db.query(SignalTransform).count()),
        ("signals", db.query(Signal).count()),
        ("  — shocks", db.query(Signal).filter(Signal.is_shock == True).count()),
        ("propagation_results", db.query(PropagationResult).count()),
        ("  — pending", db.query(PropagationResult).filter(PropagationResult.status == "pending").count()),
        ("  — monitoring", db.query(PropagationResult).filter(PropagationResult.status == "monitoring").count()),
        ("  — completed", db.query(PropagationResult).filter(PropagationResult.status == "completed").count()),
        ("causal_test_results", db.query(CausalTestResult).count()),
    ]
    for name, count in counts:
        print(f"  {name:<30} {count}")
    print()


def cmd_events(db, args):
    q = db.query(Event)
    if args.type:
        q = q.filter(Event.type == args.type)
    q = q.order_by(Event.timestamp.desc()).limit(args.limit)
    rows_data = q.all()
    print(f"\n=== Events (showing {len(rows_data)}) ===")
    rows = [
        (e.id[:8], fmt_dt(e.timestamp), e.type, e.subtype or "—", e.description[:50])
        for e in rows_data
    ]
    print_table(rows, ["ID (short)", "Timestamp", "Type", "Subtype", "Description"])
    print()


def cmd_transforms(db, args):
    transforms = db.query(SignalTransform).order_by(SignalTransform.created_at).all()
    print(f"\n=== Signal Transforms ({len(transforms)}) ===")
    rows = [
        (t.id[:8], t.name[:30], ", ".join(t.event_types), t.transform_expression[:35], t.node_mapping, t.unit or "—")
        for t in transforms
    ]
    print_table(rows, ["ID", "Name", "Event Types", "Expression", "Node", "Unit"])
    print()


def cmd_signals(db, args):
    q = db.query(Signal)
    if args.shock:
        q = q.filter(Signal.is_shock == True)
    q = q.order_by(Signal.timestamp.desc()).limit(args.limit)
    sigs = q.all()
    print(f"\n=== Signals (showing {len(sigs)}, shocks_only={args.shock}) ===")
    rows = [
        (
            s.id[:8],
            fmt_dt(s.timestamp),
            s.event_id[:8],
            s.transform_id[:8],
            fmt_float(s.value),
            fmt_float(s.z_score, 2) if s.z_score is not None else "—",
            "SHOCK" if s.is_shock else "ok",
        )
        for s in sigs
    ]
    print_table(rows, ["ID", "Timestamp", "Event", "Transform", "Value", "Z-score", "Status"])
    print()


def cmd_propagation(db, args):
    q = db.query(PropagationResult)
    if args.status:
        q = q.filter(PropagationResult.status == args.status)
    q = q.order_by(PropagationResult.created_at.desc()).limit(args.limit)
    props = q.all()
    print(f"\n=== Propagation Results (showing {len(props)}) ===")
    rows = [
        (
            p.id[:8],
            p.source_node,
            p.target_node,
            p.expected_lag_weeks,
            p.expected_direction,
            fmt_float(p.actual_change, 4) if p.actual_change is not None else "—",
            "yes" if p.propagation_matched else ("no" if p.propagation_matched is False else "—"),
            p.status,
        )
        for p in props
    ]
    print_table(rows, ["ID", "Source", "Target", "Lag(wk)", "Direction", "Actual Δ", "Matched", "Status"])
    print()


def cmd_causal(db, args):
    results = db.query(CausalTestResult).order_by(CausalTestResult.created_at.desc()).limit(args.limit).all()
    print(f"\n=== Causal Test Results (showing {len(results)}) ===")
    rows = [
        (
            r.id[:8],
            r.transform_id[:8],
            r.target_node,
            r.num_events_used,
            fmt_float(r.coefficient),
            fmt_float(r.p_value),
            fmt_float(r.r_squared, 3),
            r.confidence_level or "—",
            "yes" if r.is_causal else "no",
        )
        for r in results
    ]
    print_table(rows, ["ID", "Transform", "Target Node", "N", "Coeff", "p-value", "R²", "Confidence", "Causal"])
    print()


def cmd_event_detail(db, args):
    event_id = args.id
    # Support short ID prefix
    event = db.query(Event).filter(Event.id.startswith(event_id)).first()
    if not event:
        print(f"No event found with ID starting with '{event_id}'")
        return

    print(f"\n=== Event: {event.id} ===")
    print(f"  Timestamp:   {fmt_dt(event.timestamp)}")
    print(f"  Type:        {event.type}")
    print(f"  Subtype:     {event.subtype or '—'}")
    print(f"  Description: {event.description}")
    print(f"  Tags:        {event.tags}")
    print(f"  Raw Data:    {json.dumps(event.raw_data, indent=4)}")
    print(f"  Created:     {fmt_dt(event.created_at)}")

    sigs = db.query(Signal).filter(Signal.event_id == event.id).all()
    if sigs:
        print(f"\n  Signals ({len(sigs)}):")
        for s in sigs:
            transform = db.query(SignalTransform).filter(SignalTransform.id == s.transform_id).first()
            name = transform.name if transform else s.transform_id[:8]
            shock_flag = " [SHOCK]" if s.is_shock else ""
            print(f"    [{name}]  value={fmt_float(s.value)}  z={fmt_float(s.z_score, 2)}{shock_flag}")
    else:
        print("\n  No signals computed yet. Run: POST /signals/compute/{event_id}")
    print()


def cmd_shell(db, args):
    """Drop into an interactive sqlite3 shell."""
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    print(f"\nOpening sqlite3 shell for: {db_path}")
    print("Tip: .tables | .schema events | SELECT * FROM events LIMIT 5;\n")
    subprocess.run(["sqlite3", db_path])


def main():
    parser = argparse.ArgumentParser(
        description="Tremor database CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Summary counts for all tables")

    p_events = subparsers.add_parser("events", help="List events")
    p_events.add_argument("--type", help="Filter by event type")
    p_events.add_argument("--limit", type=int, default=20)

    subparsers.add_parser("transforms", help="List signal transforms")

    p_sigs = subparsers.add_parser("signals", help="List signals")
    p_sigs.add_argument("--shock", action="store_true", help="Show only shocks")
    p_sigs.add_argument("--limit", type=int, default=20)

    p_prop = subparsers.add_parser("propagation", help="List propagation results")
    p_prop.add_argument("--status", help="Filter by status (pending/monitoring/completed)")
    p_prop.add_argument("--limit", type=int, default=20)

    p_causal = subparsers.add_parser("causal", help="List causal test results")
    p_causal.add_argument("--limit", type=int, default=20)

    p_event = subparsers.add_parser("event", help="Show full event detail")
    p_event.add_argument("id", help="Event ID (or short prefix)")

    subparsers.add_parser("shell", help="Open interactive sqlite3 shell")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    init_db()
    db = SessionLocal()
    try:
        dispatch = {
            "status": cmd_status,
            "events": cmd_events,
            "transforms": cmd_transforms,
            "signals": cmd_signals,
            "propagation": cmd_propagation,
            "causal": cmd_causal,
            "event": cmd_event_detail,
            "shell": cmd_shell,
        }
        dispatch[args.command](db, args)
    finally:
        db.close()


if __name__ == "__main__":
    main()
