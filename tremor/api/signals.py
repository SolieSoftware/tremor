from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from tremor.core.signal_factory import compute_signals_for_event
from tremor.models.database import Event, Signal, SignalTransform, get_db
from tremor.models.schemas import SignalResponse, SignalTransformCreate, SignalTransformResponse

router = APIRouter(prefix="/signals", tags=["Signals"])


# --- Transforms ---


@router.post("/transforms", response_model=SignalTransformResponse)
def create_transform(body: SignalTransformCreate, db: Session = Depends(get_db)):
    transform = SignalTransform(
        name=body.name,
        description=body.description,
        event_types=body.event_types,
        transform_expression=body.transform_expression,
        unit=body.unit,
        node_mapping=body.node_mapping,
        threshold_sd=body.threshold_sd,
    )
    db.add(transform)
    db.commit()
    db.refresh(transform)
    return transform


@router.get("/transforms", response_model=list[SignalTransformResponse])
def list_transforms(db: Session = Depends(get_db)):
    return db.query(SignalTransform).all()


@router.get("/transforms/{transform_id}", response_model=SignalTransformResponse)
def get_transform(transform_id: str, db: Session = Depends(get_db)):
    transform = db.query(SignalTransform).filter(SignalTransform.id == transform_id).first()
    if not transform:
        raise HTTPException(status_code=404, detail="Transform not found")
    return transform


@router.delete("/transforms/{transform_id}")
def delete_transform(transform_id: str, db: Session = Depends(get_db)):
    transform = db.query(SignalTransform).filter(SignalTransform.id == transform_id).first()
    if not transform:
        raise HTTPException(status_code=404, detail="Transform not found")
    db.delete(transform)
    db.commit()
    return {"detail": "Transform deleted"}


# --- Signals ---


@router.post("/compute/{event_id}", response_model=list[SignalResponse])
def compute_signals(event_id: str, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return compute_signals_for_event(event, db)


@router.get("", response_model=list[SignalResponse])
def list_signals(
    event_id: Optional[str] = Query(None),
    transform_id: Optional[str] = Query(None),
    is_shock: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Signal)
    if event_id:
        q = q.filter(Signal.event_id == event_id)
    if transform_id:
        q = q.filter(Signal.transform_id == transform_id)
    if is_shock is not None:
        q = q.filter(Signal.is_shock == is_shock)
    return q.order_by(Signal.timestamp.desc()).offset(offset).limit(limit).all()
