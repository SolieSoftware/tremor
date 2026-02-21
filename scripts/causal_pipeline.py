"""Causal pipeline CLI.

Runs the full sequence for a given transform:
  1. Check how many signals exist (feasibility)
  2. Show where the transform maps in the causal network (downstream nodes)
  3. Run the event study (OLS dose-response) against each downstream target
  4. Print a summary of causal findings

Usage:
    python scripts/causal_pipeline.py --transform "CPI Surprise"
    python scripts/causal_pipeline.py --transform "CPI Surprise" --pre-window 5 --post-window 10
    python scripts/causal_pipeline.py --list-transforms
    python scripts/causal_pipeline.py --feasibility
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tremor.models.database import (
    CausalTestResult, Signal, SignalTransform, SessionLocal, init_db,
)
from tremor.causal.network import load_network, get_downstream_nodes, get_edge_info, get_all_edges
from tremor.causal.event_study import run_event_study
from tremor.config import settings


def fmt_float(v, d=4):
    return f"{v:.{d}f}" if v is not None else "—"


def stars(p):
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def load_graph():
    """Load causal network, trying GraphML then CSV."""
    for path in [settings.CAUSAL_NETWORK_PATH, settings.GRANGER_RESULTS_PATH]:
        try:
            load_network(path)
            edges = get_all_edges()
            if edges:
                print(f"  Causal network loaded from: {path} ({len(edges)} edges)")
                return True
        except FileNotFoundError:
            continue
    print("  WARNING: No causal network file found. Graph tests will be skipped.")
    print(f"  Expected at: {settings.CAUSAL_NETWORK_PATH} or {settings.GRANGER_RESULTS_PATH}")
    return False


def cmd_list_transforms(db):
    transforms = db.query(SignalTransform).order_by(SignalTransform.name).all()
    print(f"\n=== Available Transforms ({len(transforms)}) ===")
    for t in transforms:
        count = db.query(Signal).filter(Signal.transform_id == t.id).count()
        feasible = "✓" if count >= settings.MIN_EVENTS_FOR_CAUSAL_TEST else "✗"
        print(f"  [{feasible}] {t.name:<30} node={t.node_mapping:<20} signals={count}")
    print(f"\n  (✓ = enough signals for event study, min={settings.MIN_EVENTS_FOR_CAUSAL_TEST})")


def cmd_feasibility(db):
    transforms = db.query(SignalTransform).all()
    print(f"\n=== Event Study Feasibility (min events = {settings.MIN_EVENTS_FOR_CAUSAL_TEST}) ===")
    for t in transforms:
        count = db.query(Signal).filter(Signal.transform_id == t.id).count()
        status = "READY" if count >= settings.MIN_EVENTS_FOR_CAUSAL_TEST else f"need {settings.MIN_EVENTS_FOR_CAUSAL_TEST - count} more"
        print(f"  {t.name:<35} {count:>3} signals  →  {status}")


def cmd_run_pipeline(db, args):
    # ── Find transform ────────────────────────────────────────────
    transform = (
        db.query(SignalTransform)
        .filter(SignalTransform.name == args.transform)
        .first()
    )
    if not transform:
        # Try partial match
        transform = (
            db.query(SignalTransform)
            .filter(SignalTransform.name.ilike(f"%{args.transform}%"))
            .first()
        )
    if not transform:
        print(f"ERROR: Transform '{args.transform}' not found.")
        print("Run --list-transforms to see available transforms.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Causal Pipeline: {transform.name}")
    print(f"{'='*65}")
    print(f"  Expression:   {transform.transform_expression}")
    print(f"  Source node:  {transform.node_mapping}")
    print(f"  Event types:  {', '.join(transform.event_types)}")

    # ── Signal count ──────────────────────────────────────────────
    signal_count = db.query(Signal).filter(Signal.transform_id == transform.id).count()
    print(f"  Signals in DB: {signal_count}")

    if signal_count < settings.MIN_EVENTS_FOR_CAUSAL_TEST:
        print(f"\n  INSUFFICIENT DATA: need {settings.MIN_EVENTS_FOR_CAUSAL_TEST} signals, have {signal_count}.")
        print(f"  Ingest more events: python scripts/ingest.py fred --series CPIAUCSL --limit 24 --compute-signals")
        sys.exit(1)

    # ── Load causal network ───────────────────────────────────────
    print(f"\n--- Step 1: Causal Network ---")
    has_network = load_graph()

    source_node = transform.node_mapping
    downstream = get_downstream_nodes(source_node) if has_network else []

    if has_network:
        print(f"  Source node '{source_node}' has {len(downstream)} downstream targets:")
        for target in downstream:
            edge = get_edge_info(source_node, target)
            if edge:
                print(f"    → {target:<25} lag={edge.get('lag',1)}wk  "
                      f"F={fmt_float(edge.get('f_statistic'),1)}  "
                      f"p={fmt_float(edge.get('p_value'),3)}")
        if not downstream:
            print(f"  Node '{source_node}' has no outgoing edges in the network.")
            print("  The event study will still run but without network-guided targets.")

    # ── Determine targets ─────────────────────────────────────────
    # Use downstream nodes from graph; fall back to known nodes if graph empty
    all_nodes = ["d_treasury_10y", "d_fed_funds", "d_credit_spread", "d_vix", "sp500_ret"]
    targets = downstream if downstream else [n for n in all_nodes if n != source_node]

    if args.target:
        targets = [args.target]
        print(f"  Overriding targets: [{args.target}]")

    # ── Run event study per target ────────────────────────────────
    print(f"\n--- Step 2: Event Study (OLS dose-response) ---")
    print(f"  Windows: pre={args.pre_window}d  post={args.post_window}d  gap={args.gap}d")
    print(f"  Significance level: {args.significance}")
    print()

    results = []
    for target in targets:
        print(f"  Testing {source_node} → {target} ...", end="", flush=True)
        try:
            result = run_event_study(
                transform_id=transform.id,
                target_node=target,
                pre_window_days=args.pre_window,
                post_window_days=args.post_window,
                exclude_overlapping=args.exclude_overlapping,
                gap_days=args.gap,
                overlap_buffer_days=args.overlap_buffer,
                significance_level=args.significance,
                db=db,
            )
            results.append((target, result, None))
            flag = f"[{result.confidence_level.upper()}]" if result.confidence_level else "[none]"
            print(f" {flag} p={result.p_value:.3f} coeff={result.coefficient:+.4f}")
        except ValueError as e:
            results.append((target, None, str(e)))
            print(f" SKIPPED — {e}")
        except Exception as e:
            results.append((target, None, str(e)))
            print(f" ERROR — {e}")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  RESULTS: {transform.name}  ({signal_count} events)")
    print(f"{'='*65}")
    print(f"  {'Target':<25} {'Coeff':>8} {'p-val':>7} {'R²':>6} {'Sig':>4} {'Causal?':<10} Confidence")
    print(f"  {'-'*25}  {'-'*7}  {'-'*6}  {'-'*5}  {'-'*3}  {'-'*9}  {'-'*10}")

    for target, result, error in results:
        if error:
            print(f"  {target:<25}  {'—':>8}  {'—':>7}  {'—':>6}  {'—':>4}  {'ERROR':<10} {error[:30]}")
        else:
            sig = stars(result.p_value)
            causal = "YES" if result.is_causal else "no"
            print(
                f"  {target:<25} "
                f"{result.coefficient:>+8.4f}  "
                f"{result.p_value:>7.3f}  "
                f"{result.r_squared:>6.3f}  "
                f"{sig:>4}  "
                f"{causal:<10} "
                f"{result.confidence_level or 'none'}"
            )

    print(f"\n  Significance: *** p<0.01  ** p<0.05  * p<0.10")

    # ── Placebo summary ───────────────────────────────────────────
    successful = [(t, r) for t, r, e in results if r is not None]
    if successful:
        print(f"\n--- Placebo Tests ---")
        print(f"  {'Target':<25} {'Pre-drift p':>12} {'Pre-drift':>10} {'Zero-surp p':>12} {'Zero-surp':>10}")
        print(f"  {'-'*25}  {'-'*11}  {'-'*9}  {'-'*11}  {'-'*9}")
        for target, r in successful:
            pre_p = fmt_float(r.placebo_pre_drift_pvalue, 3)
            pre_pass = "PASS" if r.placebo_pre_drift_pvalue and r.placebo_pre_drift_pvalue > args.significance else "FAIL"
            zero_p = fmt_float(r.placebo_zero_surprise_pvalue, 3)
            zero_pass = "PASS" if r.placebo_zero_surprise_pvalue and r.placebo_zero_surprise_pvalue > args.significance else ("N/A" if r.placebo_zero_surprise_pvalue is None else "FAIL")
            print(f"  {target:<25}  {pre_p:>11}  {pre_pass:>9}  {zero_p:>11}  {zero_pass:>9}")

    print(f"\n  Results saved to DB. View with:")
    print(f"    python scripts/db_cli.py causal")
    print(f"    python scripts/causal_pipeline.py --feasibility")


def main():
    parser = argparse.ArgumentParser(
        description="Tremor causal pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--transform", help="Transform name to test (e.g. 'CPI Surprise')")
    parser.add_argument("--target", help="Override target node (default: all downstream nodes)")
    parser.add_argument("--pre-window", type=int, default=5, help="Pre-event window in trading days (default 5)")
    parser.add_argument("--post-window", type=int, default=5, help="Post-event window in trading days (default 5)")
    parser.add_argument("--gap", type=int, default=1, help="Gap days around event (default 1)")
    parser.add_argument("--overlap-buffer", type=int, default=10, help="Days buffer for overlap exclusion (default 10)")
    parser.add_argument("--no-exclude-overlapping", dest="exclude_overlapping", action="store_false", default=True)
    parser.add_argument("--significance", type=float, default=0.05, help="Significance level (default 0.05)")
    parser.add_argument("--list-transforms", action="store_true", help="List all transforms and signal counts")
    parser.add_argument("--feasibility", action="store_true", help="Show event study feasibility for all transforms")

    args = parser.parse_args()

    init_db()
    db = SessionLocal()
    try:
        if args.list_transforms:
            cmd_list_transforms(db)
        elif args.feasibility:
            cmd_feasibility(db)
        elif args.transform:
            cmd_run_pipeline(db, args)
        else:
            parser.print_help()
    finally:
        db.close()


if __name__ == "__main__":
    main()
