from simpleeval import simple_eval, EvalWithCompoundTypes

from sqlalchemy.orm import Session

from tremor.core.shock_detector import detect_shock
from tremor.models.database import Event, Signal, SignalTransform


def safe_eval_expression(expression: str, raw_data: dict) -> float:
    """Safely evaluate a transform expression against event raw_data.

    Only allows basic arithmetic on values from raw_data.
    """
    evaluator = EvalWithCompoundTypes(names=raw_data)
    return float(evaluator.eval(expression))


def get_matching_transforms(event_type: str, db: Session) -> list[SignalTransform]:
    """Find all transforms whose event_types include the given event type."""
    transforms = db.query(SignalTransform).all()
    return [t for t in transforms if event_type in t.event_types]


def compute_signals_for_event(event: Event, db: Session) -> list[Signal]:
    """Compute signals for an event using all matching transforms."""
    transforms = get_matching_transforms(event.type, db)
    signals = []

    for transform in transforms:
        try:
            value = safe_eval_expression(transform.transform_expression, event.raw_data)
        except Exception:
            continue

        historical_values = [
            s.value
            for s in db.query(Signal)
            .filter(Signal.transform_id == transform.id)
            .all()
        ]

        z_score, is_shock = detect_shock(value, historical_values, transform.threshold_sd)

        signal = Signal(
            event_id=event.id,
            transform_id=transform.id,
            timestamp=event.timestamp,
            value=value,
            z_score=z_score,
            is_shock=is_shock,
        )
        db.add(signal)
        signals.append(signal)

    db.commit()
    for s in signals:
        db.refresh(s)
    return signals
