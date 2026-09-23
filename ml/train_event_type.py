import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import ADASYN

DATA_DIR = os.environ.get(
    "CAT_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "../data")
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Extract time features from timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(9).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_night"] = (df["hour_of_day"] < 6) | (df["hour_of_day"] >= 20)
    df["is_night"] = df["is_night"].astype(int)

    # Encode categoricals
    df["seatbelt_status"] = df["seatbelt_status"].map({"Fastened": 0, "Unfastened": 1}).fillna(0)
    df["risk_level"] = df["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2}).fillna(0)
    df["shift_type"] = df["shift_type"].map({"day": 0, "night": 1}).fillna(0)
    df["resolved"] = df["resolved"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)

    weather_enc = LabelEncoder()
    df["weather"] = weather_enc.fit_transform(df["weather"].astype(str))

    machine_enc = LabelEncoder()
    df["machine_type"] = machine_enc.fit_transform(df["machine_type"].astype(str))

    # Derived features that help discriminate event types
    df["is_close_worker"] = (df["worker_distance_m"] < 3.0).astype(int)
    df["is_high_speed"] = (df["speed_kmh"] > 8.0).astype(int)
    df["is_high_temp"] = (df["engine_temp_c"] > 95.0).astype(int)
    df["speed_x_distance"] = df["speed_kmh"] * df["worker_distance_m"]
    df["temp_speed_ratio"] = df["engine_temp_c"] / (df["speed_kmh"] + 0.1)
    df["risk_x_speed"] = df["risk_level"] * df["speed_kmh"]
    df["risk_x_distance"] = df["risk_level"] * df["worker_distance_m"]
    df["response_time_sec"] = df["response_time_sec"].fillna(df["response_time_sec"].median())

    return df, weather_enc, machine_enc


FEATURE_COLS = [
    "speed_kmh", "worker_distance_m", "engine_temp_c",
    "seatbelt_status", "risk_level", "weather", "machine_type",
    "shift_type", "hour_of_day", "day_of_week", "is_night",
    "resolved", "response_time_sec",
    "is_close_worker", "is_high_speed", "is_high_temp",
    "speed_x_distance", "temp_speed_ratio",
    "risk_x_speed", "risk_x_distance"
]


def train():
    print("=== Training Event Type Classifier (Improved) ===")

    df = pd.read_csv(os.path.join(DATA_DIR, "safety_events.csv"))
    df, weather_enc, machine_enc = engineer_features(df)

    event_enc = LabelEncoder()
    df["event_label"] = event_enc.fit_transform(df["event_type"].astype(str))

    print("  Class distribution:")
    for label, name in enumerate(event_enc.classes_):
        count = (df["event_label"] == label).sum()
        print(f"    {name:20s}: {count}")

    X = df[FEATURE_COLS].fillna(0)
    y = df["event_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ADASYN — better than SMOTE for extreme class imbalance
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=3)
        X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
        print(f"  After ADASYN: {len(X_train_res)} training samples")
    except Exception as e:
        print(f"  ADASYN fallback to raw data: {e}")
        X_train_res, y_train_res = X_train, y_train

    # XGBoost multi-class — better than RF for this problem
    # scale weights inversely proportional to class frequency
    class_counts = np.bincount(y_train_res)
    sample_weights = np.array([1.0 / class_counts[c] for c in y_train_res])
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0.1,
        objective="multi:softprob",
        num_class=len(event_enc.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_res, y_train_res,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Overall Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(classification_report(y_test, y_pred, target_names=event_enc.classes_))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importance
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top features: {top}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "event_type_model.pkl"))
    joblib.dump(event_enc, os.path.join(MODELS_DIR, "event_type_encoder.pkl"))
    joblib.dump(weather_enc, os.path.join(MODELS_DIR, "event_weather_enc.pkl"))
    joblib.dump(machine_enc, os.path.join(MODELS_DIR, "event_machine_enc.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "event_type_features.pkl"))
    print("  Saved: event_type_model.pkl\n")

    return acc


if __name__ == "__main__":
    train()
