from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from tremor.causal.event_study import run_event_study
from tremor.config import settings
from tremor.models.database import CausalTestResult, Signal, SignalTransform, get_db
from tremor.models.schemas import (
    CausalTestRequest,
    CausalTestResponse,
    CausalTestSummary,
    EventStudyDetail,
    PlaceboResults,
    RegressionResults,
)

router = APIRouter(prefix="/causal-tests", tags=["Causal Tests"])


def _build_response(result: CausalTestResult) -> CausalTestResponse:
    """Build a full CausalTestResponse from a database result."""
    regression = RegressionResults(
        coefficient=result.coefficient,
        std_error=result.std_error,
        t_statistic=result.t_statistic,
        p_value=result.p_value,
        r_squared=result.r_squared,
        conf_interval_lower=result.conf_interval_lower,
        conf_interval_upper=result.conf_interval_upper,
        intercept=result.intercept,
        intercept_p_value=result.intercept_p_value,
        num_observations=result.num_events_used,
    )

    placebo = PlaceboResults(
        pre_drift_coefficient=result.placebo_pre_drift_coeff,
        pre_drift_p_value=result.placebo_pre_drift_pvalue,
        pre_drift_passed=(
            result.placebo_pre_drift_pvalue > settings.CAUSAL_SIGNIFICANCE_LEVEL
            if result.placebo_pre_drift_pvalue is not None
            else None
        ),
        zero_surprise_coefficient=result.placebo_zero_surprise_coeff,
        zero_surprise_p_value=result.placebo_zero_surprise_pvalue,
        zero_surprise_passed=(
            result.placebo_zero_surprise_pvalue > settings.CAUSAL_SIGNIFICANCE_LEVEL
            if result.placebo_zero_surprise_pvalue is not None
            else None
        ),
    )

    event_details = [
        EventStudyDetail(**detail) for detail in (result.event_details or [])
    ]

    return CausalTestResponse(
        id=result.id,
        transform_id=result.transform_id,
        target_node=result.target_node,
        pre_window_days=result.pre_window_days,
        post_window_days=result.post_window_days,
        gap_days=result.gap_days,
        num_events=result.num_events,
        num_events_used=result.num_events_used,
        num_events_excluded=result.num_events_excluded,
        regression=regression,
        placebo=placebo,
        is_causal=result.is_causal,
        confidence_level=result.confidence_level,
        event_details=event_details,
        created_at=result.created_at,
    )


@router.post("/run", response_model=CausalTestResponse)
def run_causal_test(body: CausalTestRequest, db: Session = Depends(get_db)):
    transform = (
        db.query(SignalTransform)
        .filter(SignalTransform.id == body.transform_id)
        .first()
    )
    if not transform:
        raise HTTPException(status_code=404, detail="Transform not found")

    try:
        result = run_event_study(
            transform_id=body.transform_id,
            target_node=body.target_node,
            pre_window_days=body.pre_window_days,
            post_window_days=body.post_window_days,
            gap_days=body.gap_days,
            exclude_overlapping=body.exclude_overlapping,
            overlap_buffer_days=body.overlap_buffer_days,
            significance_level=body.confidence_level,
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _build_response(result)


@router.get("", response_model=list[CausalTestSummary])
def list_causal_tests(
    transform_id: Optional[str] = Query(None),
    target_node: Optional[str] = Query(None),
    is_causal: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(CausalTestResult)
    if transform_id:
        q = q.filter(CausalTestResult.transform_id == transform_id)
    if target_node:
        q = q.filter(CausalTestResult.target_node == target_node)
    if is_causal is not None:
        q = q.filter(CausalTestResult.is_causal == is_causal)
    return q.order_by(CausalTestResult.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/feasibility")
def check_feasibility(
    min_events: int = Query(settings.MIN_EVENTS_FOR_CAUSAL_TEST, ge=1),
    db: Session = Depends(get_db),
):
    transforms = db.query(SignalTransform).all()
    results = []
    for t in transforms:
        count = db.query(Signal).filter(Signal.transform_id == t.id).count()
        results.append({
            "transform_id": t.id,
            "transform_name": t.name,
            "target_node": t.node_mapping,
            "num_events": count,
            "feasible": count >= min_events,
        })
    return results


@router.get("/{test_id}", response_model=CausalTestResponse)
def get_causal_test(test_id: str, db: Session = Depends(get_db)):
    result = (
        db.query(CausalTestResult)
        .filter(CausalTestResult.id == test_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Causal test result not found")
    return _build_response(result)


@router.delete("/{test_id}")
def delete_causal_test(test_id: str, db: Session = Depends(get_db)):
    result = (
        db.query(CausalTestResult)
        .filter(CausalTestResult.id == test_id)
        .first()
    )
    if not result:
        raise HTTPException(status_code=404, detail="Causal test result not found")
    db.delete(result)
    db.commit()
    return {"detail": "Causal test result deleted"}
