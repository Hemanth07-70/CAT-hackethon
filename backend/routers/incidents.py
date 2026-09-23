import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from database import get_db
from db_models import SafetyIncident
from schemas import IncidentCreate, IncidentResponse

router = APIRouter(prefix="/incidents", tags=["Safety Incidents"])


@router.get(
    "/",
    response_model=List[IncidentResponse],
    summary="Get safety incidents",
    description="Filter by machine, operator, risk level, or resolved status."
)
def get_incidents(
    machine_id: Optional[str] = Query(None),
    operator_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    resolved: Optional[bool] = Query(None),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db)
):
    q = db.query(SafetyIncident)
    if machine_id:
        q = q.filter(SafetyIncident.machine_id == machine_id)
    if operator_id:
        q = q.filter(SafetyIncident.operator_id == operator_id)
    if risk_level:
        q = q.filter(SafetyIncident.risk_level == risk_level)
    if resolved is not None:
        q = q.filter(SafetyIncident.resolved == resolved)
    return q.order_by(SafetyIncident.id.desc()).limit(limit).all()


@router.post(
    "/",
    response_model=IncidentResponse,
    status_code=201,
    summary="Log a safety incident"
)
def create_incident(req: IncidentCreate, db: Session = Depends(get_db)):
    incident = SafetyIncident(
        event_id=f"EVT{uuid.uuid4().hex[:6].upper()}",
        **req.model_dump()
    )
    db.add(incident)
    db.commit()
    db.refresh(incident)
    return incident


@router.patch(
    "/{event_id}/resolve",
    response_model=IncidentResponse,
    summary="Mark incident as resolved"
)
def resolve_incident(event_id: str, db: Session = Depends(get_db)):
    incident = db.query(SafetyIncident).filter(SafetyIncident.event_id == event_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    incident.resolved = True
    db.commit()
    db.refresh(incident)
    return incident
