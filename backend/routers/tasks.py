import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from database import get_db
from db_models import Task
from schemas import TaskCreate, TaskUpdate, TaskResponse
from ml_service import predict_task_time

router = APIRouter(prefix="/tasks", tags=["Tasks"])


@router.get(
    "/",
    response_model=List[TaskResponse],
    summary="Get all tasks",
    description="Returns the daily task list. Filter by operator_id or status."
)
def get_tasks(
    operator_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(Task)
    if operator_id:
        q = q.filter(Task.operator_id == operator_id)
    if status:
        q = q.filter(Task.status == status)
    return q.order_by(Task.created_at.desc()).all()


@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get task by ID"
)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post(
    "/",
    response_model=TaskResponse,
    status_code=201,
    summary="Create a new task",
    description="Creates a task and automatically runs task time prediction."
)
def create_task(req: TaskCreate, db: Session = Depends(get_db)):
    predicted = None
    try:
        result = predict_task_time(req.model_dump())
        predicted = result["predicted_time_min"]
    except Exception:
        pass

    task = Task(
        task_id=f"T{uuid.uuid4().hex[:6].upper()}",
        predicted_time_min=predicted,
        **req.model_dump()
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@router.put(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Update task status or actual time"
)
def update_task(task_id: str, req: TaskUpdate, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    for field, value in req.model_dump(exclude_none=True).items():
        setattr(task, field, value)
    db.commit()
    db.refresh(task)
    return task


@router.delete(
    "/{task_id}",
    summary="Delete a task"
)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
    return {"message": f"Task {task_id} deleted"}
