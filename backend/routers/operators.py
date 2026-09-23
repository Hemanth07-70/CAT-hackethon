from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from db_models import OperatorProfile
from schemas import OperatorResponse

router = APIRouter(prefix="/operators", tags=["Operators"])


@router.get(
    "/",
    response_model=List[OperatorResponse],
    summary="Get all operator profiles"
)
def get_operators(
    skill_level: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(OperatorProfile)
    if skill_level:
        q = q.filter(OperatorProfile.skill_level == skill_level)
    return q.order_by(OperatorProfile.safety_score.desc()).all()


@router.get(
    "/{operator_id}",
    response_model=OperatorResponse,
    summary="Get a single operator profile"
)
def get_operator(operator_id: str, db: Session = Depends(get_db)):
    op = db.query(OperatorProfile).filter(OperatorProfile.operator_id == operator_id).first()
    if not op:
        raise HTTPException(status_code=404, detail="Operator not found")
    return op
