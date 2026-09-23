import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

_task_model = None
_task_encoders = None
_task_features = None

_anomaly_model = None
_anomaly_weather_enc = None
_anomaly_features = None

_safety_model = None
_safety_features = None

_event_model = None
_event_enc = None
_event_weather_enc = None
_event_machine_enc = None
_event_features = None


def load_all_models():
    global _task_model, _task_encoders, _task_features
    global _anomaly_model, _anomaly_weather_enc, _anomaly_features
    global _safety_model, _safety_features
    global _event_model, _event_enc, _event_weather_enc, _event_machine_enc, _event_features

    _task_model = joblib.load(os.path.join(MODELS_DIR, "task_time_model.pkl"))
    _task_encoders = joblib.load(os.path.join(MODELS_DIR, "task_time_encoders.pkl"))
    _task_features = joblib.load(os.path.join(MODELS_DIR, "task_time_features.pkl"))

    _anomaly_model = joblib.load(os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    _anomaly_weather_enc = joblib.load(os.path.join(MODELS_DIR, "anomaly_weather_enc.pkl"))
    _anomaly_features = joblib.load(os.path.join(MODELS_DIR, "anomaly_features.pkl"))

    _safety_model = joblib.load(os.path.join(MODELS_DIR, "safety_alert_model.pkl"))
    _safety_features = joblib.load(os.path.join(MODELS_DIR, "safety_alert_features.pkl"))

    _event_model = joblib.load(os.path.join(MODELS_DIR, "event_type_model.pkl"))
    _event_enc = joblib.load(os.path.join(MODELS_DIR, "event_type_encoder.pkl"))
    _event_weather_enc = joblib.load(os.path.join(MODELS_DIR, "event_weather_enc.pkl"))
    _event_machine_enc = joblib.load(os.path.join(MODELS_DIR, "event_machine_enc.pkl"))
    _event_features = joblib.load(os.path.join(MODELS_DIR, "event_type_features.pkl"))

    print("All ML models loaded.")


def _skill_encode(skill: str) -> int:
    return {"Beginner": 0, "Intermediate": 1, "Expert": 2}.get(skill, 1)


def _shift_encode(shift: str) -> int:
    return 0 if shift == "day" else 1


def predict_task_time(data: dict) -> dict:
    row = {}
    for col in ["task_type", "weather", "soil_condition", "machine_type"]:
        enc = _task_encoders[col]
        val = data.get(col, enc.classes_[0])
        try:
            row[col] = enc.transform([val])[0]
        except ValueError:
            row[col] = 0

    row["operator_skill_level"] = _skill_encode(data.get("operator_skill_level", "Intermediate"))
    row["shift_type"] = _shift_encode(data.get("shift_type", "day"))

    numeric_fields = [
        "operator_experience_years", "engine_hours", "fuel_level_pct",
        "payload_tonnes", "temperature_c", "distance_m", "idle_time_pct",
        "previous_task_time_min", "estimated_time_min", "hour_of_day",
        "month", "safety_score", "fuel_efficiency_score",
        "precision_score", "idle_management_score"
    ]
    defaults = {
        "operator_experience_years": 5, "engine_hours": 3000, "fuel_level_pct": 75,
        "payload_tonnes": 10, "temperature_c": 28, "distance_m": 150,
        "idle_time_pct": 20, "previous_task_time_min": 60, "estimated_time_min": 60,
        "hour_of_day": 9, "month": 6, "safety_score": 75,
        "fuel_efficiency_score": 75, "precision_score": 75, "idle_management_score": 75
    }
    for field in numeric_fields:
        row[field] = data.get(field, defaults[field])

    X = pd.DataFrame([row])[_task_features]
    predicted = float(_task_model.predict(X)[0])
    return {"predicted_time_min": round(predicted, 1)}


def predict_anomaly(data: dict) -> dict:
    row = {}
    try:
        weather_val = _anomaly_weather_enc.transform([data.get("weather", "Sunny")])[0]
    except ValueError:
        weather_val = 0

    row["weather"] = weather_val
    row["shift_type"] = _shift_encode(data.get("shift_type", "day"))

    numeric_fields = {
        "idle_pct": 40, "speed_kmh": 5, "rpm": 1400, "engine_temp_c": 80,
        "fuel_level_pct": 75, "working_fuel_rate_lph": 10, "idle_fuel_rate_lph": 3,
        "blended_fuel_rate_lph": 7, "speed_variance": 0.5, "acceleration_events": 1,
        "worker_distance_m": 15, "hour_of_day": 9, "shift_elapsed_min": 60
    }
    for field, default in numeric_fields.items():
        row[field] = data.get(field, default)

    X = pd.DataFrame([row])[_anomaly_features]
    prediction = int(_anomaly_model.predict(X)[0])
    probability = float(_anomaly_model.predict_proba(X)[0][1])
    return {
        "is_anomaly": bool(prediction),
        "anomaly_probability": round(probability, 4),
        "status": "ANOMALY DETECTED" if prediction else "Normal"
    }


def predict_safety_alert(data: dict) -> dict:
    row = {
        "seatbelt_status": 0 if data.get("seatbelt_status") == "Fastened" else 1,
        "risk_level": {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(data.get("risk_level", "LOW"), 0),
        "shift_type": _shift_encode(data.get("shift_type", "day")),
    }
    numeric_fields = {
        "worker_distance_m": 15, "speed_kmh": 5, "acceleration_events": 1,
        "idle_pct": 40, "engine_temp_c": 80, "blended_fuel_rate_lph": 7,
        "speed_variance": 0.5, "hour_of_day": 9
    }
    for field, default in numeric_fields.items():
        row[field] = data.get(field, default)

    X = pd.DataFrame([row])[_safety_features]
    prediction = int(_safety_model.predict(X)[0])
    probability = float(_safety_model.predict_proba(X)[0][1])
    return {
        "alert_triggered": bool(prediction),
        "alert_probability": round(probability, 4),
        "status": "ALERT" if prediction else "Safe"
    }


def predict_event_type(data: dict) -> dict:
    speed  = data.get("speed_kmh", 5)
    distance = data.get("worker_distance_m", 15)
    temp   = data.get("engine_temp_c", 80)
    risk   = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(data.get("risk_level", "LOW"), 0)
    hour   = data.get("hour_of_day", 9)
    fuel   = data.get("fuel_rate_lph", 8.0)
    accel  = data.get("acceleration_events", 1)

    row = {
        "seatbelt_status":     0 if data.get("seatbelt_status") == "Fastened" else 1,
        "risk_level":          risk,
        "shift_type":          _shift_encode(data.get("shift_type", "day")),
        "speed_kmh":           speed,
        "worker_distance_m":   distance,
        "engine_temp_c":       temp,
        "fuel_rate_lph":       fuel,
        "acceleration_events": accel,
        "hour_of_day":         hour,
        "day_of_week":         data.get("day_of_week", 0),
        "is_night":            1 if hour < 6 or hour >= 20 else 0,
        "resolved":            data.get("resolved", 0),
        "response_time_sec":   data.get("response_time_sec", 6.0),
        "is_close_worker":     1 if distance < 3.0 else 0,
        "is_high_speed":       1 if speed > 10.0 else 0,
        "is_high_temp":        1 if temp > 98.0 else 0,
        "is_high_fuel":        1 if fuel > 17.0 else 0,
        "is_erratic":          1 if accel >= 5 else 0,
        "speed_x_distance":    speed * distance,
        "temp_speed_ratio":    temp / (speed + 0.1),
        "fuel_x_accel":        fuel * accel,
        "risk_x_speed":        risk * speed,
        "risk_x_distance":     risk * distance,
    }
    try:
        row["weather"] = _event_weather_enc.transform([data.get("weather", "Sunny")])[0]
    except ValueError:
        row["weather"] = 0
    try:
        row["machine_type"] = _event_machine_enc.transform([data.get("machine_type", "Dozer")])[0]
    except ValueError:
        row["machine_type"] = 0

    X = pd.DataFrame([row])[_event_features]
    prediction = int(_event_model.predict(X)[0])
    probabilities = _event_model.predict_proba(X)[0]
    event_name = _event_enc.inverse_transform([prediction])[0]

    return {
        "event_type": event_name,
        "confidence": round(float(probabilities.max()), 4),
        "all_probabilities": {
            cls: round(float(p), 4)
            for cls, p in zip(_event_enc.classes_, probabilities)
        }
    }
