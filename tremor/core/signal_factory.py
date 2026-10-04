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
    """Compute signals for an event using all matching transforms.

    Idempotent: a transform that already produced a signal for this event
    returns the existing signal rather than creating a duplicate.
    """
    transforms = get_matching_transforms(event.type, db)
    signals = []
    new_signals = []

    for transform in transforms:
        existing = (
            db.query(Signal)
            .filter(Signal.event_id == event.id, Signal.transform_id == transform.id)
            .first()
        )
        if existing:
            signals.append(existing)
            continue

        try:
            value = safe_eval_expression(transform.transform_expression, event.raw_data)
        except Exception:
            continue

        # Only signals from earlier events form the baseline, so backfilled
        # history doesn't let later surprises leak into an earlier z-score
        historical_values = [
            s.value
            for s in db.query(Signal)
            .filter(Signal.transform_id == transform.id, Signal.timestamp < event.timestamp)
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
        new_signals.append(signal)

    db.commit()
    for s in new_signals:
        db.refresh(s)
    return signals
