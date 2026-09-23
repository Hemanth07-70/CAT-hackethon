import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

from preprocessor import load_telemetry_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

FEATURE_COLS = [
    "seatbelt_status", "worker_distance_m", "speed_kmh",
    "acceleration_events", "idle_pct", "risk_level",
    "engine_temp_c", "blended_fuel_rate_lph", "speed_variance",
    "hour_of_day", "shift_type"
]


def check_overfit(model, X_tr, y_tr, X_v, y_v):
    tr = accuracy_score(y_tr, model.predict(X_tr))
    vl = accuracy_score(y_v,  model.predict(X_v))
    gap = tr - vl
    status = "OVERFIT ⚠️" if gap > 0.05 else "OK ✅"
    print(f"  Acc Train={tr:.4f}  Val={vl:.4f}  Gap={gap:.4f}  → {status}")


def train():
    print("=== Training Safety Alert Classifier ===")
    df, weather_enc = load_telemetry_data()

    X = df[FEATURE_COLS].fillna(0)
    y = df["safety_alert"]
    print(f"  Alert=0:{(y==0).sum()}  Alert=1:{(y==1).sum()}")

    # 70 / 15 / 15 split stratified
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    print(f"  Split → Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("\n  Overfitting Check:")
    check_overfit(model, X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n  ── FINAL TEST RESULTS ──")
    print(classification_report(y_test, y_pred, target_names=["No Alert", "Alert"]))
    print(f"  ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {top}")

    cv = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print(f"  5-Fold CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,        os.path.join(MODELS_DIR, "safety_alert_model.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "safety_alert_features.pkl"))
    print("  Saved: safety_alert_model.pkl\n")


if __name__ == "__main__":
    train()
