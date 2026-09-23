from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func
from database import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, unique=True, index=True)
    operator_id = Column(String, index=True)
    machine_id = Column(String, index=True)
    machine_type = Column(String)
    task_type = Column(String)
    weather = Column(String)
    soil_condition = Column(String)
    status = Column(String, default="pending")
    estimated_time_min = Column(Float)
    predicted_time_min = Column(Float, nullable=True)
    actual_time_min = Column(Float, nullable=True)
    scheduled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class TelemetryLog(Base):
    __tablename__ = "telemetry_logs"

    id = Column(Integer, primary_key=True, index=True)
    machine_id = Column(String, index=True)
    operator_id = Column(String, index=True)
    timestamp = Column(String)
    speed_kmh = Column(Float)
    rpm = Column(Integer)
    engine_temp_c = Column(Float)
    fuel_level_pct = Column(Float)
    idle_pct = Column(Float)
    worker_distance_m = Column(Float)
    seatbelt_status = Column(String)
    safety_alert = Column(Boolean, default=False)
    risk_level = Column(String)
    anomaly_detected = Column(Boolean, default=False)
    anomaly_probability = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class SafetyIncident(Base):
    __tablename__ = "safety_incidents"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String, unique=True, index=True)
    machine_id = Column(String, index=True)
    operator_id = Column(String, index=True)
    event_type = Column(String)
    risk_level = Column(String)
    speed_kmh = Column(Float)
    worker_distance_m = Column(Float)
    seatbelt_status = Column(String)
    resolved = Column(Boolean, default=False)
    response_time_sec = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    occurred_at = Column(String)
    created_at = Column(DateTime, server_default=func.now())


class OperatorProfile(Base):
    __tablename__ = "operator_profiles"

    id = Column(Integer, primary_key=True, index=True)
    operator_id = Column(String, unique=True, index=True)
    skill_level = Column(String)
    experience_years = Column(Float)
    safety_score = Column(Integer)
    fuel_efficiency_score = Column(Integer)
    precision_score = Column(Integer)
    idle_management_score = Column(Integer)
    response_time_score = Column(Integer)
    weakest_skill = Column(String)
    shifts_completed = Column(Integer)
    avg_idle_pct = Column(Float)
