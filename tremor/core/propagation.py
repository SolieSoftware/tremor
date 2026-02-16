from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from tremor.causal.baselines import get_expected_direction, get_expected_response
from tremor.causal.network import get_downstream_nodes, get_edge_info
from tremor.config import settings
from tremor.market_data.fetcher import fetch_node_data
from tremor.models.database import PropagationResult, Signal, SignalTransform


def create_propagation_monitors(signal: Signal, db: Session) -> list[PropagationResult]:
    """Create propagation monitoring records for a shock signal.

    Looks up the signal's node_mapping, finds downstream nodes, and creates
    a PropagationResult for each with expected lag/direction from the causal network.
    """
    transform = db.query(SignalTransform).filter(SignalTransform.id == signal.transform_id).first()
    if not transform:
        return []

    source_node = transform.node_mapping
    downstream = get_downstream_nodes(source_node)
    results = []

    for target_node in downstream:
        edge = get_edge_info(source_node, target_node)
        lag_weeks = edge.get("lag", 1) if edge else 1
        direction = get_expected_direction(source_node, target_node) or "positive"
        magnitude = get_expected_response(source_node, target_node, lag_weeks)

        buffer = settings.PROPAGATION_BUFFER_WEEKS
        monitored_from = signal.timestamp
        monitored_until = signal.timestamp + timedelta(weeks=lag_weeks + buffer)

        result = PropagationResult(
            signal_id=signal.id,
            source_node=source_node,
            target_node=target_node,
            expected_lag_weeks=lag_weeks,
            expected_direction=direction,
            expected_magnitude=magnitude,
            status="monitoring",
            monitored_from=monitored_from,
            monitored_until=monitored_until,
        )
        db.add(result)
        results.append(result)

    db.commit()
    for r in results:
        db.refresh(r)
    return results


def check_propagation(propagation_id: str, db: Session) -> Optional[PropagationResult]:
    """Check whether a predicted propagation has occurred.

    Pulls market data for the target variable over the monitoring window
    and compares against expected direction and magnitude.
    """
    result = db.query(PropagationResult).filter(PropagationResult.id == propagation_id).first()
    if not result:
        return None

    now = datetime.now(timezone.utc)

    def _ensure_aware(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    monitored_until = _ensure_aware(result.monitored_until) if result.monitored_until else None
    monitored_from = _ensure_aware(result.monitored_from)
    end_date = min(now, monitored_until) if monitored_until else now

    try:
        data = fetch_node_data(result.target_node, monitored_from, end_date)
    except Exception:
        return result

    if data is None or data.empty:
        if monitored_until and now > monitored_until:
            result.status = "no_response"
            db.commit()
            db.refresh(result)
        return result

    actual_change = float(data.iloc[-1] - data.iloc[0]) if len(data) > 1 else 0.0
    result.actual_change = actual_change

    weeks_elapsed = (end_date - monitored_from).days / 7
    result.actual_lag_weeks = max(1, int(weeks_elapsed))

    direction_matched = (
        (result.expected_direction == "positive" and actual_change > 0)
        or (result.expected_direction == "negative" and actual_change < 0)
    )
    result.propagation_matched = direction_matched

    if monitored_until and now >= monitored_until:
        result.status = "completed"
    else:
        result.status = "monitoring"

    db.commit()
    db.refresh(result)
    return result
