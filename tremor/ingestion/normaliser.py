"""Normaliser: converts EventPayload → EventCreate.

This is the single translation layer between the ingestion world and the
Tremor API/DB world. All ingesters produce EventPayload; this module
produces EventCreate ready to be posted to POST /events.
"""

from tremor.ingestion.base import EventPayload
from tremor.models.schemas import EventCreate


def normalise(payload: EventPayload) -> EventCreate:
    """Convert an EventPayload into an EventCreate schema object."""
    return EventCreate(
        timestamp=payload.timestamp,
        type=payload.event_type,
        subtype=payload.event_subtype,
        description=payload.description,
        tags=payload.tags,
        raw_data=payload.to_raw_data(),
    )


def normalise_many(payloads: list[EventPayload]) -> list[EventCreate]:
    return [normalise(p) for p in payloads]
