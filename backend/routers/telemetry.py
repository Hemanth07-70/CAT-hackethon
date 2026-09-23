from typing import List, Optional
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database import get_db
from db_models import TelemetryLog
from schemas import TelemetryInput, TelemetryResponse
from ml_service import predict_anomaly, predict_safety_alert

router = APIRouter(prefix="/telemetry", tags=["Telemetry"])


@router.post(
    "/",
    response_model=TelemetryResponse,
    status_code=201,
    summary="Ingest telemetry reading",
    description="Accepts a real-time machine telemetry snapshot. Runs anomaly detection and safety alert prediction automatically."
)
def ingest(req: TelemetryInput, db: Session = Depends(get_db)):
    data = req.model_dump()

    anomaly_result = predict_anomaly(data)
    safety_result = predict_safety_alert(data)

    log = TelemetryLog(
        machine_id=req.machine_id,
        operator_id=req.operator_id,
        timestamp=req.timestamp,
        speed_kmh=req.speed_kmh,
        rpm=req.rpm,
        engine_temp_c=req.engine_temp_c,
        fuel_level_pct=req.fuel_level_pct,
        idle_pct=req.idle_pct,
        worker_distance_m=req.worker_distance_m,
        seatbelt_status=req.seatbelt_status,
        risk_level=req.risk_level,
        safety_alert=safety_result["alert_triggered"],
        anomaly_detected=anomaly_result["is_anomaly"],
        anomaly_probability=anomaly_result["anomaly_probability"],
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


@router.get(
    "/",
    response_model=List[TelemetryResponse],
    summary="Get telemetry logs",
    description="Fetch telemetry logs filtered by machine or operator."
)
def get_logs(
    machine_id: Optional[str] = Query(None),
    operator_id: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    q = db.query(TelemetryLog)
    if machine_id:
        q = q.filter(TelemetryLog.machine_id == machine_id)
    if operator_id:
        q = q.filter(TelemetryLog.operator_id == operator_id)
    return q.order_by(TelemetryLog.id.desc()).limit(limit).all()


@router.get(
    "/anomalies",
    response_model=List[TelemetryResponse],
    summary="Get all anomaly detections"
)
def get_anomalies(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    return (
        db.query(TelemetryLog)
        .filter(TelemetryLog.anomaly_detected == True)
        .order_by(TelemetryLog.id.desc())
        .limit(limit)
        .all()
    )
