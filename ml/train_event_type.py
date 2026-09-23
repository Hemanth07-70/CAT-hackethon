import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

DATA_DIR   = os.environ.get("CAT_DATA_DIR", os.path.join(os.path.dirname(__file__), "../data"))
MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def engineer_features(df: pd.DataFrame):
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"]  = df["timestamp"].dt.hour.fillna(9).astype(int)
    df["day_of_week"]  = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_night"]     = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 20)).astype(int)

    df["seatbelt_status"] = df["seatbelt_status"].map({"Fastened": 0, "Unfastened": 1}).fillna(0)
    df["risk_level"]      = df["risk_level"].map({"LOW": 0, "MEDIUM": 1, "HIGH": 2}).fillna(0)
    df["shift_type"]      = df["shift_type"].map({"day": 0, "night": 1}).fillna(0)
    df["resolved"]        = df["resolved"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0)

    weather_enc = LabelEncoder()
    df["weather"] = weather_enc.fit_transform(df["weather"].astype(str))

    machine_enc = LabelEncoder()
    df["machine_type"] = machine_enc.fit_transform(df["machine_type"].astype(str))

    df["fuel_rate_lph"]       = df.get("fuel_rate_lph", 8.0).fillna(8.0)
    df["acceleration_events"] = df.get("acceleration_events", 2).fillna(2)
    df["response_time_sec"]   = df["response_time_sec"].fillna(df["response_time_sec"].median())

    # Derived features
    df["is_close_worker"]    = (df["worker_distance_m"] < 3.0).astype(int)
    df["is_high_speed"]      = (df["speed_kmh"] > 10.0).astype(int)
    df["is_high_temp"]       = (df["engine_temp_c"] > 98.0).astype(int)
    df["is_high_fuel"]       = (df["fuel_rate_lph"] > 17.0).astype(int)
    df["is_erratic"]         = (df["acceleration_events"] >= 5).astype(int)
    df["speed_x_distance"]   = df["speed_kmh"] * df["worker_distance_m"]
    df["temp_speed_ratio"]   = df["engine_temp_c"] / (df["speed_kmh"] + 0.1)
    df["fuel_x_accel"]       = df["fuel_rate_lph"] * df["acceleration_events"]
    df["risk_x_speed"]       = df["risk_level"] * df["speed_kmh"]
    df["risk_x_distance"]    = df["risk_level"] * df["worker_distance_m"]

    return df, weather_enc, machine_enc


FEATURE_COLS = [
    # Raw features
    "speed_kmh", "worker_distance_m", "engine_temp_c",
    "fuel_rate_lph", "acceleration_events",
    "seatbelt_status", "risk_level", "weather", "machine_type",
    "shift_type", "hour_of_day", "day_of_week", "is_night",
    "resolved", "response_time_sec",
    # Engineered features
    "is_close_worker", "is_high_speed", "is_high_temp",
    "is_high_fuel", "is_erratic",
    "speed_x_distance", "temp_speed_ratio",
    "fuel_x_accel", "risk_x_speed", "risk_x_distance",
]


def check_overfitting(model, X_train, y_train, X_val, y_val, label=""):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc   = accuracy_score(y_val,   model.predict(X_val))
    gap       = train_acc - val_acc
    status    = "OVERFIT ⚠️" if gap > 0.05 else "OK ✅"
    print(f"  {label} | Train={train_acc:.4f}  Val={val_acc:.4f}  Gap={gap:.4f}  → {status}")
    return gap


def train():
    print("=== Training Event Type Classifier (Full Pipeline) ===")

    # ── Load augmented dataset ──────────────────────────────────────────────
    aug_path = os.path.join(DATA_DIR, "safety_events_augmented.csv")
    df = pd.read_csv(aug_path)
    df, weather_enc, machine_enc = engineer_features(df)

    event_enc = LabelEncoder()
    df["event_label"] = event_enc.fit_transform(df["event_type"].astype(str))

    print(f"  Total samples: {len(df)}")
    print("  Class distribution:")
    for label, name in enumerate(event_enc.classes_):
        count = (df["event_label"] == label).sum()
        print(f"    {name:20s}: {count}")

    X = df[FEATURE_COLS].fillna(0).values
    y = df["event_label"].values

    # ── Train / Validate / Test split (70 / 15 / 15) ──────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
        # 0.176 of 0.85 ≈ 15% of total
    )

    print(f"\n  Split → Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    # ── Train XGBoost with early stopping on validation ────────────────────
    model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.2,
        reg_alpha=0.5,
        reg_lambda=3.0,
        objective="multi:softprob",
        num_class=len(event_enc.classes_),
        eval_metric="mlogloss",
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    best_iter = model.best_iteration
    print(f"  Best iteration (early stopping): {best_iter}")

    # ── Overfitting check ──────────────────────────────────────────────────
    print("\n  Overfitting Check:")
    gap = check_overfitting(model, X_train, y_train, X_val, y_val, "Train vs Val")
    check_overfitting(model, X_train, y_train, X_test, y_test, "Train vs Test")

    # ── Cross-validation on full train+val ────────────────────────────────
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    cv_model = xgb.XGBClassifier(
        n_estimators=best_iter or 200, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        objective="multi:softprob", num_class=len(event_enc.classes_),
        eval_metric="mlogloss", n_jobs=-1
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X_trainval, y_trainval, cv=skf, scoring="accuracy")
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Final test evaluation ──────────────────────────────────────────────
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  ── FINAL TEST RESULTS ──────────────────────────────")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Weighted F1   : {test_f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=event_enc.classes_))
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:7]
    print(f"\n  Top 7 features:")
    for feat, imp in top:
        print(f"    {feat:25s}: {imp:.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,       os.path.join(MODELS_DIR, "event_type_model.pkl"))
    joblib.dump(event_enc,   os.path.join(MODELS_DIR, "event_type_encoder.pkl"))
    joblib.dump(weather_enc, os.path.join(MODELS_DIR, "event_weather_enc.pkl"))
    joblib.dump(machine_enc, os.path.join(MODELS_DIR, "event_machine_enc.pkl"))
    joblib.dump(FEATURE_COLS,os.path.join(MODELS_DIR, "event_type_features.pkl"))
    print("\n  Saved: event_type_model.pkl")

    return {"test_accuracy": round(test_acc, 4), "weighted_f1": round(test_f1, 4)}


if __name__ == "__main__":
    train()
