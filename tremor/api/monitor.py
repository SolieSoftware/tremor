from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from tremor.causal import network as causal_network_module
from tremor.causal.network import get_all_edges
from tremor.core.propagation import check_propagation, create_propagation_monitors
from tremor.models.database import PropagationResult, Signal, get_db
from tremor.models.schemas import (
    EdgeInfo,
    NetworkStatusResponse,
    PropagationResponse,
    ShockResponse,
    SignalTransformResponse,
)

router = APIRouter(prefix="/monitor", tags=["Causal Monitor"])


@router.get("/shocks", response_model=list[ShockResponse])
def list_shocks(
    source_node: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: Session = Depends(get_db),
):
    q = (
        db.query(Signal)
        .filter(Signal.is_shock.is_(True))
        .options(joinedload(Signal.event), joinedload(Signal.transform))
    )

    if start_date:
        q = q.filter(Signal.timestamp >= start_date)
    if end_date:
        q = q.filter(Signal.timestamp <= end_date)

    shocks = q.order_by(Signal.timestamp.desc()).all()

    if source_node:
        shocks = [s for s in shocks if s.transform.node_mapping == source_node]

    if status:
        filtered = []
        for s in shocks:
            props = (
                db.query(PropagationResult)
                .filter(PropagationResult.signal_id == s.id)
                .all()
            )
            if any(p.status == status for p in props):
                filtered.append(s)
        shocks = filtered

    return [
        ShockResponse(
            signal=s,
            event=s.event,
            transform=s.transform,
        )
        for s in shocks
    ]


@router.get("/shocks/{signal_id}/propagation", response_model=list[PropagationResponse])
def get_shock_propagation(signal_id: str, db: Session = Depends(get_db)):
    signal = db.query(Signal).filter(Signal.id == signal_id, Signal.is_shock.is_(True)).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Shock signal not found")

    results = (
        db.query(PropagationResult)
        .filter(PropagationResult.signal_id == signal_id)
        .all()
    )

    if not results:
        results = create_propagation_monitors(signal, db)

    return results


@router.post("/shocks/{signal_id}/check", response_model=list[PropagationResponse])
def check_shock_propagation(signal_id: str, db: Session = Depends(get_db)):
    signal = db.query(Signal).filter(Signal.id == signal_id, Signal.is_shock.is_(True)).first()
    if not signal:
        raise HTTPException(status_code=404, detail="Shock signal not found")

    results = (
        db.query(PropagationResult)
        .filter(PropagationResult.signal_id == signal_id)
        .all()
    )

    if not results:
        results = create_propagation_monitors(signal, db)

    checked = []
    for r in results:
        updated = check_propagation(r.id, db)
        if updated:
            checked.append(updated)

    return checked


@router.get("/network", response_model=NetworkStatusResponse)
def get_network():
    edges = get_all_edges()
    return NetworkStatusResponse(
        nodes=list(causal_network_module.causal_network.nodes()),
        edges=[
            EdgeInfo(
                source=e["source"],
                target=e["target"],
                f_statistic=e.get("f_statistic"),
                lag=int(e["lag"]) if "lag" in e else None,
                p_value=e.get("p_value"),
            )
            for e in edges
        ],
        total_nodes=causal_network_module.causal_network.number_of_nodes(),
        total_edges=causal_network_module.causal_network.number_of_edges(),
    )


@router.get("/network/health")
def network_health(db: Session = Depends(get_db)):
    """Compare recent propagation results against baselines to detect regime changes."""
    completed = (
        db.query(PropagationResult)
        .filter(PropagationResult.status == "completed")
        .all()
    )

    if not completed:
        return {"status": "no_data", "message": "No completed propagation results yet"}

    total = len(completed)
    matched = sum(1 for r in completed if r.propagation_matched is True)
    unmatched = sum(1 for r in completed if r.propagation_matched is False)

    match_rate = matched / total if total > 0 else 0.0

    edge_stats: dict[str, dict] = {}
    for r in completed:
        key = f"{r.source_node} → {r.target_node}"
        if key not in edge_stats:
            edge_stats[key] = {"total": 0, "matched": 0}
        edge_stats[key]["total"] += 1
        if r.propagation_matched:
            edge_stats[key]["matched"] += 1

    for key, stats in edge_stats.items():
        stats["match_rate"] = stats["matched"] / stats["total"] if stats["total"] > 0 else 0.0

    return {
        "status": "healthy" if match_rate > 0.5 else "degraded",
        "overall_match_rate": match_rate,
        "total_completed": total,
        "total_matched": matched,
        "total_unmatched": unmatched,
        "edge_stats": edge_stats,
    }
