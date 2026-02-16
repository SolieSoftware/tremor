import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from tremor.config import settings


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    subtype: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    description: Mapped[str] = mapped_column(String, nullable=False)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    raw_data: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    signals: Mapped[list["Signal"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class SignalTransform(Base):
    __tablename__ = "signal_transforms"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    event_types: Mapped[list] = mapped_column(JSON, nullable=False)
    transform_expression: Mapped[str] = mapped_column(String, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    node_mapping: Mapped[str] = mapped_column(String, nullable=False)
    threshold_sd: Mapped[float] = mapped_column(Float, default=2.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    signals: Mapped[list["Signal"]] = relationship(
        back_populates="transform", cascade="all, delete-orphan"
    )


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id: Mapped[str] = mapped_column(String, ForeignKey("events.id"), nullable=False)
    transform_id: Mapped[str] = mapped_column(
        String, ForeignKey("signal_transforms.id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    z_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_shock: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    event: Mapped["Event"] = relationship(back_populates="signals")
    transform: Mapped["SignalTransform"] = relationship(back_populates="signals")
    propagation_results: Mapped[list["PropagationResult"]] = relationship(
        back_populates="signal", cascade="all, delete-orphan"
    )


class PropagationResult(Base):
    __tablename__ = "propagation_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    signal_id: Mapped[str] = mapped_column(String, ForeignKey("signals.id"), nullable=False)
    source_node: Mapped[str] = mapped_column(String, nullable=False)
    target_node: Mapped[str] = mapped_column(String, nullable=False)
    expected_lag_weeks: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_direction: Mapped[str] = mapped_column(String, nullable=False)
    expected_magnitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_lag_weeks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    propagation_matched: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    monitored_from: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    monitored_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    signal: Mapped["Signal"] = relationship(back_populates="propagation_results")


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
