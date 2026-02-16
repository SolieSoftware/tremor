"""Pre-load default signal transforms into the database."""

import sys
from pathlib import Path

# Add project root to path so tremor package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tremor.models.database import SessionLocal, SignalTransform, init_db

DEFAULT_TRANSFORMS = [
    {
        "name": "Fed Rate Surprise",
        "description": "Difference between actual and expected federal funds rate",
        "event_types": ["fed_announcement"],
        "transform_expression": "actual_rate - expected_rate",
        "node_mapping": "d_fed_funds",
        "unit": "percent",
        "threshold_sd": 2.0,
    },
    {
        "name": "CPI Surprise",
        "description": "CPI actual vs expected — surprises hit Treasury yields",
        "event_types": ["economic_data"],
        "transform_expression": "actual_cpi - expected_cpi",
        "node_mapping": "d_treasury_10y",
        "unit": "percent",
        "threshold_sd": 2.0,
    },
    {
        "name": "Earnings Beat",
        "description": "Actual EPS minus expected EPS",
        "event_types": ["earnings"],
        "transform_expression": "actual_eps - expected_eps",
        "node_mapping": "sp500_ret",
        "unit": "dollars",
        "threshold_sd": 2.0,
    },
    {
        "name": "VIX Spike",
        "description": "Change in VIX around a geopolitical event",
        "event_types": ["geopolitical"],
        "transform_expression": "vix_after - vix_before",
        "node_mapping": "d_vix",
        "unit": "index_points",
        "threshold_sd": 2.0,
    },
    {
        "name": "Credit Stress",
        "description": "Change in credit spreads around stress events",
        "event_types": ["geopolitical", "economic_data"],
        "transform_expression": "spread_after - spread_before",
        "node_mapping": "d_credit_spread",
        "unit": "bps",
        "threshold_sd": 2.0,
    },
    {
        "name": "NFP Surprise",
        "description": "Non-farm payrolls actual vs expected — surprises move yields",
        "event_types": ["economic_data"],
        "transform_expression": "actual_nfp - expected_nfp",
        "node_mapping": "d_treasury_10y",
        "unit": "thousands",
        "threshold_sd": 2.0,
    },
    {
        "name": "GDP Surprise",
        "description": "GDP actual vs expected — surprises move equities",
        "event_types": ["economic_data"],
        "transform_expression": "actual_gdp - expected_gdp",
        "node_mapping": "sp500_ret",
        "unit": "percent",
        "threshold_sd": 2.0,
    },
    {
        "name": "Treasury Yield Shock",
        "description": "Change in 10y Treasury yield around announcements",
        "event_types": ["fed_announcement", "economic_data"],
        "transform_expression": "yield_after - yield_before",
        "node_mapping": "d_treasury_10y",
        "unit": "bps",
        "threshold_sd": 2.0,
    },
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        for t in DEFAULT_TRANSFORMS:
            existing = db.query(SignalTransform).filter(SignalTransform.name == t["name"]).first()
            if existing:
                print(f"  Skipping '{t['name']}' (already exists)")
                continue
            transform = SignalTransform(**t)
            db.add(transform)
            print(f"  Added '{t['name']}'")
        db.commit()
        print(f"\nDone. {db.query(SignalTransform).count()} transforms in database.")
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding default signal transforms...")
    seed()
