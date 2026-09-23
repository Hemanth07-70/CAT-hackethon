from fastapi import APIRouter, HTTPException
from schemas import (
    TaskTimePredictRequest, TaskTimePredictResponse,
    AnomalyDetectRequest, AnomalyDetectResponse,
    SafetyAlertRequest, SafetyAlertResponse,
    EventTypeRequest, EventTypeResponse,
)
from ml_service import (
    predict_task_time, predict_anomaly,
    predict_safety_alert, predict_event_type,
)

router = APIRouter(prefix="/predict", tags=["ML Predictions"])


@router.post(
    "/task-time",
    response_model=TaskTimePredictResponse,
    summary="Predict how long a task will take",
    description="Uses XGBoost regression trained on historical task data to predict actual completion time."
)
def task_time(req: TaskTimePredictRequest):
    try:
        result = predict_task_time(req.model_dump())
        return TaskTimePredictResponse(
            predicted_time_min=result["predicted_time_min"],
            estimated_time_min=req.estimated_time_min,
            variance_min=round(result["predicted_time_min"] - req.estimated_time_min, 1)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/anomaly",
    response_model=AnomalyDetectResponse,
    summary="Detect unusual machine behavior",
    description="XGBoost classifier detects anomalies like excessive idling or erratic operation."
)
def anomaly(req: AnomalyDetectRequest):
    try:
        result = predict_anomaly(req.model_dump())
        return AnomalyDetectResponse(machine_id=req.machine_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/safety-alert",
    response_model=SafetyAlertResponse,
    summary="Predict safety alert risk",
    description="Random Forest classifier predicts if current conditions will trigger a safety alert."
)
def safety_alert(req: SafetyAlertRequest):
    try:
        result = predict_safety_alert(req.model_dump())
        return SafetyAlertResponse(
            machine_id=req.machine_id,
            operator_id=req.operator_id,
            **result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/event-type",
    response_model=EventTypeResponse,
    summary="Classify machine event type",
    description="Classifies the type of safety event: normal, proximity_breach, erratic_accel, temp_spike, high_fuel, overspeed."
)
def event_type(req: EventTypeRequest):
    try:
        result = predict_event_type(req.model_dump())
        return EventTypeResponse(machine_id=req.machine_id, **result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
