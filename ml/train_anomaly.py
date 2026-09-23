import os
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocessor import load_telemetry_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "idle_pct", "speed_kmh", "rpm", "engine_temp_c", "fuel_level_pct",
    "working_fuel_rate_lph", "idle_fuel_rate_lph", "blended_fuel_rate_lph",
    "speed_variance", "acceleration_events", "worker_distance_m",
    "hour_of_day", "shift_elapsed_min", "shift_type", "weather"
]


def train():
    print("=== Training Anomaly Detection Model ===")
    df, weather_enc = load_telemetry_data()

    X = df[FEATURE_COLS].fillna(0)
    y = df["anomaly_label"]   # 0=normal, 1=anomaly

    normal_count = (y == 0).sum()
    anomaly_count = (y == 1).sum()
    scale_pos_weight = normal_count / anomaly_count
    print(f"  Normal: {normal_count} | Anomaly: {anomaly_count} | scale_pos_weight: {scale_pos_weight:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    joblib.dump(weather_enc, os.path.join(MODELS_DIR, "anomaly_weather_enc.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "anomaly_features.pkl"))
    print("  Saved: anomaly_model.pkl\n")


if __name__ == "__main__":
    train()
