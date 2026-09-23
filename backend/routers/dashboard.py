from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
from db_models import Task, TelemetryLog, SafetyIncident, OperatorProfile
from schemas import DashboardSummary

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "/summary",
    response_model=DashboardSummary,
    summary="Get dashboard summary statistics",
    description="Returns counts for tasks, incidents, anomalies, and alerts for the frontend dashboard."
)
def get_summary(db: Session = Depends(get_db)):
    return DashboardSummary(
        total_tasks=db.query(Task).count(),
        tasks_pending=db.query(Task).filter(Task.status == "pending").count(),
        tasks_in_progress=db.query(Task).filter(Task.status == "in_progress").count(),
        tasks_completed=db.query(Task).filter(Task.status == "completed").count(),
        total_incidents=db.query(SafetyIncident).count(),
        unresolved_incidents=db.query(SafetyIncident).filter(SafetyIncident.resolved == False).count(),
        high_risk_incidents=db.query(SafetyIncident).filter(SafetyIncident.risk_level == "HIGH").count(),
        total_anomalies_detected=db.query(TelemetryLog).filter(TelemetryLog.anomaly_detected == True).count(),
        total_safety_alerts=db.query(TelemetryLog).filter(TelemetryLog.safety_alert == True).count(),
        active_operators=db.query(OperatorProfile).count(),
    )
