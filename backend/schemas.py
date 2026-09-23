from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


# ── Task schemas ──────────────────────────────────────────────
class TaskCreate(BaseModel):
    operator_id: str
    machine_id: str
    machine_type: str
    task_type: str
    weather: str
    soil_condition: str
    estimated_time_min: float
    scheduled_at: Optional[datetime] = None


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    actual_time_min: Optional[float] = None
    weather: Optional[str] = None


class TaskResponse(BaseModel):
    id: int
    task_id: str
    operator_id: str
    machine_id: str
    machine_type: str
    task_type: str
    weather: str
    soil_condition: str
    status: str
    estimated_time_min: float
    predicted_time_min: Optional[float]
    actual_time_min: Optional[float]
    scheduled_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Telemetry schemas ─────────────────────────────────────────
class TelemetryInput(BaseModel):
    machine_id: str
    operator_id: str
    timestamp: str
    speed_kmh: float = Field(..., ge=0)
    rpm: int = Field(..., ge=0)
    engine_temp_c: float
    fuel_level_pct: float = Field(..., ge=0, le=100)
    idle_pct: float = Field(..., ge=0, le=100)
    worker_distance_m: float = Field(..., ge=0)
    seatbelt_status: str = Field(..., pattern="^(Fastened|Unfastened)$")
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    weather: Optional[str] = "Sunny"
    shift_type: Optional[str] = "day"
    shift_elapsed_min: Optional[float] = 0
    hour_of_day: Optional[int] = 9
    working_fuel_rate_lph: Optional[float] = 10.0
    idle_fuel_rate_lph: Optional[float] = 3.0
    blended_fuel_rate_lph: Optional[float] = 7.0
    speed_variance: Optional[float] = 0.5
    acceleration_events: Optional[int] = 0


class TelemetryResponse(BaseModel):
    id: int
    machine_id: str
    operator_id: str
    timestamp: str
    safety_alert: bool
    anomaly_detected: bool
    anomaly_probability: Optional[float]
    risk_level: str

    class Config:
        from_attributes = True


# ── Prediction schemas ────────────────────────────────────────
class TaskTimePredictRequest(BaseModel):
    task_type: str = "Earth Excavation"
    weather: str = "Sunny"
    soil_condition: str = "Dry"
    operator_skill_level: str = "Intermediate"
    operator_experience_years: float = 5.0
    engine_hours: float = 3000.0
    fuel_level_pct: float = 75.0
    payload_tonnes: float = 10.0
    temperature_c: float = 28.0
    distance_m: float = 150.0
    idle_time_pct: float = 20.0
    previous_task_time_min: float = 60.0
    estimated_time_min: float = 60.0
    hour_of_day: int = 9
    shift_type: str = "day"
    month: int = 6
    machine_type: str = "Excavator"
    safety_score: Optional[int] = 75
    fuel_efficiency_score: Optional[int] = 75
    precision_score: Optional[int] = 75
    idle_management_score: Optional[int] = 75


class TaskTimePredictResponse(BaseModel):
    predicted_time_min: float
    estimated_time_min: float
    variance_min: float


class AnomalyDetectRequest(BaseModel):
    machine_id: str
    idle_pct: float
    speed_kmh: float
    rpm: int
    engine_temp_c: float
    fuel_level_pct: float
    working_fuel_rate_lph: float
    idle_fuel_rate_lph: float
    blended_fuel_rate_lph: float
    speed_variance: float
    acceleration_events: int
    worker_distance_m: float
    hour_of_day: int
    shift_elapsed_min: float
    shift_type: str = "day"
    weather: str = "Sunny"


class AnomalyDetectResponse(BaseModel):
    machine_id: str
    is_anomaly: bool
    anomaly_probability: float
    status: str


class SafetyAlertRequest(BaseModel):
    machine_id: str
    operator_id: str
    seatbelt_status: str = Field(..., pattern="^(Fastened|Unfastened)$")
    worker_distance_m: float
    speed_kmh: float
    acceleration_events: int
    idle_pct: float
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    engine_temp_c: float
    blended_fuel_rate_lph: float
    speed_variance: float
    hour_of_day: int
    shift_type: str = "day"


class SafetyAlertResponse(BaseModel):
    machine_id: str
    operator_id: str
    alert_triggered: bool
    alert_probability: float
    status: str


class EventTypeRequest(BaseModel):
    machine_id: str
    machine_type: str
    speed_kmh: float
    worker_distance_m: float
    engine_temp_c: float
    seatbelt_status: str
    risk_level: str
    weather: str = "Sunny"
    shift_type: str = "day"


class EventTypeResponse(BaseModel):
    machine_id: str
    event_type: str
    confidence: float
    all_probabilities: Dict[str, float]


# ── Incident schemas ──────────────────────────────────────────
class IncidentCreate(BaseModel):
    machine_id: str
    operator_id: str
    event_type: str
    risk_level: str
    speed_kmh: float
    worker_distance_m: float
    seatbelt_status: str
    occurred_at: str
    response_time_sec: Optional[float] = None
    notes: Optional[str] = None


class IncidentResponse(BaseModel):
    id: int
    event_id: str
    machine_id: str
    operator_id: str
    event_type: str
    risk_level: str
    resolved: bool
    occurred_at: str
    created_at: datetime

    class Config:
        from_attributes = True


# ── Operator schemas ──────────────────────────────────────────
class OperatorResponse(BaseModel):
    operator_id: str
    skill_level: str
    experience_years: float
    safety_score: int
    fuel_efficiency_score: int
    precision_score: int
    idle_management_score: int
    response_time_score: int
    weakest_skill: str
    shifts_completed: int
    avg_idle_pct: float

    class Config:
        from_attributes = True


# ── Dashboard schema ──────────────────────────────────────────
class DashboardSummary(BaseModel):
    total_tasks: int
    tasks_pending: int
    tasks_in_progress: int
    tasks_completed: int
    total_incidents: int
    unresolved_incidents: int
    high_risk_incidents: int
    total_anomalies_detected: int
    total_safety_alerts: int
    active_operators: int
