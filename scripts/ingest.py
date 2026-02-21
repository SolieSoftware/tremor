"""Tremor ingestion CLI.

Pulls data from a source, normalises it into Events, writes to DB,
and optionally computes signals immediately.

Usage:
    python scripts/ingest.py fred --series CPIAUCSL --limit 12
    python scripts/ingest.py fred --series PAYEMS --limit 6
    python scripts/ingest.py fred --series A191RL1Q225SBEA --limit 8
    python scripts/ingest.py polygon --ticker AAPL --limit 8
    python scripts/ingest.py polygon --ticker MSFT --limit 4
    python scripts/ingest.py fed           # scrape latest FOMC releases (needs ANTHROPIC_API_KEY)
    python scripts/ingest.py rss           # scrape Reuters/AP news (needs ANTHROPIC_API_KEY)
    python scripts/ingest.py whitehouse    # scrape White House briefings (needs ANTHROPIC_API_KEY)

Flags:
    --compute-signals    After writing events, compute signals for each one
    --dry-run            Print events without writing to DB
    --since YYYY-MM-DD   Only ingest events after this date

Environment variables (prefix TREMOR_ to override):
    TREMOR_FRED_API_KEY
    TREMOR_POLYGON_API_KEY
    TREMOR_ANTHROPIC_API_KEY
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tremor.config import settings
from tremor.ingestion.normaliser import normalise_many
from tremor.models.database import Event, SessionLocal, Signal, SignalTransform, init_db
from tremor.core.signal_factory import compute_signals_for_event as _compute_signals_for_event


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M UTC") if dt else "—"


def print_event(ev, idx=None):
    prefix = f"  [{idx}]" if idx is not None else "  "
    print(f"{prefix} {fmt_dt(ev.timestamp)}  [{ev.type}/{ev.subtype}]")
    print(f"       {ev.description}")
    print(f"       raw_data: {json.dumps(ev.raw_data)}")


def filter_since(payloads, since_date):
    if not since_date:
        return payloads
    cutoff = datetime.strptime(since_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return [p for p in payloads if p.timestamp >= cutoff]


def write_to_db(db, event_creates):
    """Write EventCreate objects to DB. Returns list of (Event, is_new) tuples."""
    results = []
    for ec in event_creates:
        # Deduplicate by timestamp + type + subtype
        existing = (
            db.query(Event)
            .filter(
                Event.timestamp == ec.timestamp,
                Event.type == ec.type,
                Event.subtype == ec.subtype,
            )
            .first()
        )
        if existing:
            results.append((existing, False))
            continue

        db_event = Event(
            timestamp=ec.timestamp,
            type=ec.type,
            subtype=ec.subtype,
            description=ec.description,
            tags=ec.tags,
            raw_data=ec.raw_data,
        )
        db.add(db_event)
        db.flush()  # get the ID assigned
        results.append((db_event, True))

    db.commit()
    return results


# ── ingesters ────────────────────────────────────────────────────────────────

async def run_fred(args):
    from tremor.ingestion.api.fred import FredIngester

    key = os.environ.get("TREMOR_FRED_API_KEY") or settings.FRED_API_KEY
    if not key:
        print("ERROR: TREMOR_FRED_API_KEY not set.")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        sys.exit(1)

    ingester = FredIngester(api_key=key)

    series_ids = args.series if args.series else ["CPIAUCSL", "PAYEMS", "A191RL1Q225SBEA"]
    all_payloads = []
    for sid in series_ids:
        print(f"\n  Fetching FRED series: {sid} (limit={args.limit})")
        payloads = await ingester.fetch(
            series_id=sid,
            observation_start=args.since,
            limit=args.limit,
        )
        print(f"  → {len(payloads)} observations returned")
        all_payloads.extend(payloads)

    return all_payloads


async def run_polygon(args):
    from tremor.ingestion.api.polygon import PolygonEarningsIngester

    key = os.environ.get("TREMOR_POLYGON_API_KEY") or settings.POLYGON_API_KEY
    if not key:
        print("ERROR: TREMOR_POLYGON_API_KEY not set.")
        print("  Get a free key at: https://polygon.io/dashboard")
        sys.exit(1)

    if not args.ticker:
        print("ERROR: --ticker required for polygon ingester (e.g. --ticker AAPL)")
        sys.exit(1)

    ingester = PolygonEarningsIngester(api_key=key)
    print(f"\n  Fetching Polygon earnings for {args.ticker} (limit={args.limit})")
    payloads = await ingester.fetch(ticker=args.ticker, limit=args.limit)
    print(f"  → {len(payloads)} earnings periods returned")
    return payloads


async def run_fed(args):
    from tremor.ingestion.scrapers.fed_scraper import FedScraper

    key = os.environ.get("TREMOR_ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not key:
        print("ERROR: TREMOR_ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    os.environ["ANTHROPIC_API_KEY"] = key

    scraper = FedScraper()
    print("\n  Scraping Federal Reserve FOMC calendar...")
    payloads = await scraper.fetch_recent_releases(limit=args.limit)
    print(f"  → {len(payloads)} releases scraped")
    return payloads


async def run_rss(args):
    from tremor.ingestion.scrapers.rss_scraper import RssScraper

    key = os.environ.get("TREMOR_ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not key:
        print("ERROR: TREMOR_ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    os.environ["ANTHROPIC_API_KEY"] = key

    scraper = RssScraper()
    print("\n  Scraping Reuters/AP news RSS feeds...")
    payloads = await scraper.fetch(limit=args.limit)
    print(f"  → {len(payloads)} relevant news events returned")
    return payloads


async def run_whitehouse(args):
    from tremor.ingestion.scrapers.whitehouse_scraper import WhiteHouseScraper

    key = os.environ.get("TREMOR_ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not key:
        print("ERROR: TREMOR_ANTHROPIC_API_KEY not set.")
        sys.exit(1)
    os.environ["ANTHROPIC_API_KEY"] = key

    scraper = WhiteHouseScraper()
    print("\n  Scraping White House briefings...")
    payloads = await scraper.fetch(limit=args.limit)
    print(f"  → {len(payloads)} briefings scraped")
    return payloads


# ── signal computation ────────────────────────────────────────────────────────

def compute_signals(db, event):
    """Run signal factory for an event and return computed signals."""
    try:
        return _compute_signals_for_event(event, db)
    except Exception as e:
        print(f"    WARNING: signal computation failed: {e}")
        return []


# ── main ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="Tremor ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    # FRED
    p_fred = subparsers.add_parser("fred", help="Ingest from FRED API")
    p_fred.add_argument("--series", nargs="+", help="FRED series ID(s) e.g. CPIAUCSL PAYEMS")
    p_fred.add_argument("--limit", type=int, default=12, help="Max observations per series")
    p_fred.add_argument("--since", help="Only return observations after YYYY-MM-DD")
    p_fred.add_argument("--compute-signals", action="store_true")
    p_fred.add_argument("--dry-run", action="store_true")

    # Polygon
    p_poly = subparsers.add_parser("polygon", help="Ingest earnings from Polygon.io")
    p_poly.add_argument("--ticker", required=True, help="Stock ticker e.g. AAPL")
    p_poly.add_argument("--limit", type=int, default=8, help="Number of earnings periods")
    p_poly.add_argument("--since", help="Only return events after YYYY-MM-DD")
    p_poly.add_argument("--compute-signals", action="store_true")
    p_poly.add_argument("--dry-run", action="store_true")

    # Fed scraper
    p_fed = subparsers.add_parser("fed", help="Scrape Federal Reserve FOMC releases")
    p_fed.add_argument("--limit", type=int, default=5, help="Number of recent releases")
    p_fed.add_argument("--since", help="Only return events after YYYY-MM-DD")
    p_fed.add_argument("--compute-signals", action="store_true")
    p_fed.add_argument("--dry-run", action="store_true")

    # RSS
    p_rss = subparsers.add_parser("rss", help="Scrape Reuters/AP news RSS feeds")
    p_rss.add_argument("--limit", type=int, default=10, help="Max articles to process")
    p_rss.add_argument("--since", help="Only return events after YYYY-MM-DD")
    p_rss.add_argument("--compute-signals", action="store_true")
    p_rss.add_argument("--dry-run", action="store_true")

    # White House
    p_wh = subparsers.add_parser("whitehouse", help="Scrape White House briefings")
    p_wh.add_argument("--limit", type=int, default=5, help="Number of briefings to process")
    p_wh.add_argument("--since", help="Only return events after YYYY-MM-DD")
    p_wh.add_argument("--compute-signals", action="store_true")
    p_wh.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Tremor Ingestion — source: {args.source}")
    print(f"{'='*60}")

    # ── Fetch payloads ────────────────────────────────────────────
    runners = {
        "fred": run_fred,
        "polygon": run_polygon,
        "fed": run_fed,
        "rss": run_rss,
        "whitehouse": run_whitehouse,
    }
    payloads = await runners[args.source](args)

    if not payloads:
        print("\n  No payloads returned. Nothing to ingest.")
        return

    # ── Filter by date ────────────────────────────────────────────
    if hasattr(args, "since") and args.since:
        before = len(payloads)
        payloads = filter_since(payloads, args.since)
        print(f"\n  Date filter (since {args.since}): {before} → {len(payloads)} events")

    # ── Normalise ─────────────────────────────────────────────────
    event_creates = normalise_many(payloads)
    print(f"\n  Normalised {len(event_creates)} events")

    # ── Dry run ───────────────────────────────────────────────────
    if args.dry_run:
        print("\n  [DRY RUN] Events that would be written:")
        for i, ec in enumerate(event_creates):
            print(f"\n  [{i+1}] {fmt_dt(ec.timestamp)}  [{ec.type}/{ec.subtype}]")
            print(f"       {ec.description}")
            print(f"       raw_data: {json.dumps(ec.raw_data, indent=4)}")
        print("\n  [DRY RUN] No changes written.")
        return

    # ── Write to DB ───────────────────────────────────────────────
    init_db()
    db = SessionLocal()
    try:
        print("\n  Writing to database...")
        written = write_to_db(db, event_creates)

        new_count = sum(1 for _, is_new in written if is_new)
        skip_count = sum(1 for _, is_new in written if not is_new)
        print(f"  → {new_count} new events written, {skip_count} skipped (duplicates)")

        for event, is_new in written:
            status = "NEW  " if is_new else "SKIP "
            print(f"    [{status}] {event.id[:8]}  {fmt_dt(event.timestamp)}  {event.type}/{event.subtype}")

        # ── Compute signals ───────────────────────────────────────
        if args.compute_signals:
            print("\n  Computing signals...")
            new_events = [ev for ev, is_new in written if is_new]
            for event in new_events:
                signals = compute_signals(db, event)
                if signals:
                    shocks = [s for s in signals if s.is_shock]
                    print(f"    {event.id[:8]}  → {len(signals)} signal(s), {len(shocks)} shock(s)")
                    for s in signals:
                        shock_flag = " [SHOCK]" if s.is_shock else ""
                        transform = db.query(SignalTransform).filter(SignalTransform.id == s.transform_id).first()
                        tname = transform.name if transform else s.transform_id[:8]
                        z = f"z={s.z_score:.2f}" if s.z_score is not None else "z=N/A"
                        print(f"      [{tname}]  val={s.value:.4f}  {z}{shock_flag}")
                else:
                    print(f"    {event.id[:8]}  → no matching transforms")

        # ── Final summary ─────────────────────────────────────────
        from tremor.models.database import Event as EventModel, Signal as SignalModel
        total_events = db.query(EventModel).count()
        total_signals = db.query(SignalModel).count()
        total_shocks = db.query(SignalModel).filter(SignalModel.is_shock == True).count()

        print(f"\n{'='*60}")
        print(f"  Done.")
        print(f"  DB totals:  events={total_events}  signals={total_signals}  shocks={total_shocks}")
        print(f"\n  Next steps:")
        print(f"    python scripts/db_cli.py status")
        print(f"    python scripts/db_cli.py events --type {event_creates[0].type if event_creates else 'economic_data'}")
        if not args.compute_signals:
            print(f"    Re-run with --compute-signals to run signal factory on ingested events")
        print(f"{'='*60}\n")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
