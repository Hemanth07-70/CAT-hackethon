import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from preprocessor import load_safety_events

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "speed_kmh", "worker_distance_m", "engine_temp_c",
    "seatbelt_status", "risk_level", "weather",
    "machine_type", "shift_type"
]


def train():
    print("=== Training Event Type Classifier (Multi-class) ===")
    df, event_enc, weather_enc, machine_enc = load_safety_events()

    X = df[FEATURE_COLS].fillna(0)
    y = df["event_label"]

    print("  Class distribution:")
    for label, name in enumerate(event_enc.classes_):
        print(f"    {name}: {(y == label).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_train_res)} training samples")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=event_enc.classes_))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "event_type_model.pkl"))
    joblib.dump(event_enc, os.path.join(MODELS_DIR, "event_type_encoder.pkl"))
    joblib.dump(weather_enc, os.path.join(MODELS_DIR, "event_weather_enc.pkl"))
    joblib.dump(machine_enc, os.path.join(MODELS_DIR, "event_machine_enc.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "event_type_features.pkl"))
    print("  Saved: event_type_model.pkl\n")


if __name__ == "__main__":
    train()
