from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload

from tremor.models.database import Event, get_db
from tremor.models.schemas import EventCreate, EventResponse

router = APIRouter(prefix="/events", tags=["Events"])


@router.post("", response_model=EventResponse)
def create_event(body: EventCreate, db: Session = Depends(get_db)):
    event = Event(
        timestamp=body.timestamp,
        type=body.type,
        subtype=body.subtype,
        description=body.description,
        tags=body.tags,
        raw_data=body.raw_data,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


@router.get("", response_model=list[EventResponse])
def list_events(
    type: Optional[str] = Query(None),
    subtype: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Event)
    if type:
        q = q.filter(Event.type == type)
    if subtype:
        q = q.filter(Event.subtype == subtype)
    if tag:
        q = q.filter(Event.tags.contains(tag))
    if start_date:
        q = q.filter(Event.timestamp >= start_date)
    if end_date:
        q = q.filter(Event.timestamp <= end_date)
    return q.order_by(Event.timestamp.desc()).offset(offset).limit(limit).all()


@router.get("/{event_id}", response_model=EventResponse)
def get_event(event_id: str, db: Session = Depends(get_db)):
    event = (
        db.query(Event)
        .options(joinedload(Event.signals))
        .filter(Event.id == event_id)
        .first()
    )
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@router.delete("/{event_id}")
def delete_event(event_id: str, db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(event)
    db.commit()
    return {"detail": "Event deleted"}
