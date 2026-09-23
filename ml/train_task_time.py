import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessor import load_task_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def check_overfit(model, X_tr, y_tr, X_v, y_v):
    r2_train = r2_score(y_tr, model.predict(X_tr))
    r2_val   = r2_score(y_v,  model.predict(X_v))
    gap = r2_train - r2_val
    status = "OVERFIT ⚠️" if gap > 0.05 else "OK ✅"
    print(f"  R² Train={r2_train:.4f}  Val={r2_val:.4f}  Gap={gap:.4f}  → {status}")


def train():
    print("=== Training Task Time Estimation Model ===")
    X, y, encoders, feature_cols = load_task_data()

    # 70 / 15 / 15 split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    print(f"  Split → Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        random_state=42,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    print(f"  Best iteration: {model.best_iteration}")

    print("\n  Overfitting Check:")
    check_overfit(model, X_train, y_train, X_val, y_val)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ── FINAL TEST RESULTS ──")
    print(f"  MAE  : {mae:.2f} min")
    print(f"  RMSE : {rmse:.2f} min")
    print(f"  R²   : {r2:.4f}")

    cv = cross_val_score(
        xgb.XGBRegressor(n_estimators=model.best_iteration or 200, random_state=42),
        X, y, cv=5, scoring="r2"
    )
    print(f"  5-Fold CV R²: {cv.mean():.4f} ± {cv.std():.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model,        os.path.join(MODELS_DIR, "task_time_model.pkl"))
    joblib.dump(encoders,     os.path.join(MODELS_DIR, "task_time_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "task_time_features.pkl"))
    print("  Saved: task_time_model.pkl\n")
    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


if __name__ == "__main__":
    train()
