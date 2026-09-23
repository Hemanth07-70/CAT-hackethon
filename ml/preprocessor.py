import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.environ.get(
    "CAT_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "../data")
)


def load_task_data():
    tasks = pd.read_csv(os.path.join(DATA_DIR, "task_eta_data.csv"))
    profiles = pd.read_csv(os.path.join(DATA_DIR, "operator_profiles.csv"))

    df = tasks.merge(
        profiles[["operator_id", "safety_score", "fuel_efficiency_score",
                  "precision_score", "idle_management_score"]],
        on="operator_id", how="left"
    )

    df["operator_skill_level"] = df["operator_skill_level"].map(
        {"Beginner": 0, "Intermediate": 1, "Expert": 2}
    )
    df["shift_type"] = df["shift_type"].map({"day": 0, "night": 1})

    cat_cols = ["task_type", "weather", "soil_condition", "machine_type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [
        "task_type", "weather", "soil_condition", "operator_skill_level",
        "operator_experience_years", "engine_hours", "fuel_level_pct",
        "payload_tonnes", "temperature_c", "distance_m", "idle_time_pct",
        "previous_task_time_min", "estimated_time_min", "hour_of_day",
        "shift_type", "month", "machine_type",
        "safety_score", "fuel_efficiency_score", "precision_score",
        "idle_management_score"
    ]

    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df["actual_time_min"]
    return X, y, encoders, feature_cols


def load_telemetry_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "telemetry_data.csv"))

    df["anomaly_label"] = df["anomaly_label"].map({1: 0, -1: 1})
    df["safety_alert"] = df["safety_alert"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
    df["seatbelt_status"] = df["seatbelt_status"].map({"Fastened": 0, "Unfastened": 1})
    df["risk_level"] = df["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    df["shift_type"] = df["shift_type"].map({"day": 0, "night": 1})

    weather_enc = LabelEncoder()
    df["weather"] = weather_enc.fit_transform(df["weather"].astype(str))

    return df, weather_enc


def load_safety_events():
    df = pd.read_csv(os.path.join(DATA_DIR, "safety_events.csv"))

    df["seatbelt_status"] = df["seatbelt_status"].map({"Fastened": 0, "Unfastened": 1})
    df["risk_level"] = df["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2})
    df["shift_type"] = df["shift_type"].map({"day": 0, "night": 1})
    df["resolved"] = df["resolved"].map({"True": 1, "False": 0, True: 1, False: 0})

    weather_enc = LabelEncoder()
    df["weather"] = weather_enc.fit_transform(df["weather"].astype(str))

    machine_enc = LabelEncoder()
    df["machine_type"] = machine_enc.fit_transform(df["machine_type"].astype(str))

    event_enc = LabelEncoder()
    df["event_label"] = event_enc.fit_transform(df["event_type"].astype(str))

    return df, event_enc, weather_enc, machine_enc
