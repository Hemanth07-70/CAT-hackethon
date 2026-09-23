import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from preprocessor import load_telemetry_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "seatbelt_status", "worker_distance_m", "speed_kmh",
    "acceleration_events", "idle_pct", "risk_level",
    "engine_temp_c", "blended_fuel_rate_lph", "speed_variance",
    "hour_of_day", "shift_type"
]


def train():
    print("=== Training Safety Alert Classifier ===")
    df, weather_enc = load_telemetry_data()

    X = df[FEATURE_COLS].fillna(0)
    y = df["safety_alert"]   # 0=no alert, 1=alert

    print(f"  Alert=0: {(y==0).sum()} | Alert=1: {(y==1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, target_names=["No Alert", "Alert"]))
    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {top}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "safety_alert_model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "safety_alert_features.pkl"))
    print("  Saved: safety_alert_model.pkl\n")


if __name__ == "__main__":
    train()
