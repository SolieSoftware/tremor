from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


# --- Events ---


class EventCreate(BaseModel):
    timestamp: datetime
    type: str
    subtype: Optional[str] = None
    description: str
    tags: list[str] = []
    raw_data: dict = {}


class SignalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    event_id: str
    transform_id: str
    timestamp: datetime
    value: float
    z_score: Optional[float] = None
    is_shock: bool
    created_at: datetime


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    timestamp: datetime
    type: str
    subtype: Optional[str] = None
    description: str
    tags: list[str]
    raw_data: dict
    created_at: datetime
    signals: list[SignalResponse] = []


# --- Signal Transforms ---


class SignalTransformCreate(BaseModel):
    name: str
    description: Optional[str] = None
    event_types: list[str]
    transform_expression: str
    unit: Optional[str] = None
    node_mapping: str
    threshold_sd: float = 2.0


class SignalTransformResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: Optional[str] = None
    event_types: list[str]
    transform_expression: str
    unit: Optional[str] = None
    node_mapping: str
    threshold_sd: float
    created_at: datetime


# --- Shocks ---


class ShockResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    signal: SignalResponse
    event: EventResponse
    transform: SignalTransformResponse


# --- Propagation ---


class PropagationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    signal_id: str
    source_node: str
    target_node: str
    expected_lag_weeks: int
    expected_direction: str
    expected_magnitude: Optional[float] = None
    actual_change: Optional[float] = None
    actual_lag_weeks: Optional[int] = None
    propagation_matched: Optional[bool] = None
    status: str
    monitored_from: datetime
    monitored_until: Optional[datetime] = None
    created_at: datetime


# --- Network ---


class EdgeInfo(BaseModel):
    source: str
    target: str
    f_statistic: Optional[float] = None
    lag: Optional[int] = None
    p_value: Optional[float] = None


class NetworkStatusResponse(BaseModel):
    nodes: list[str]
    edges: list[EdgeInfo]
    total_nodes: int
    total_edges: int
