import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessor import load_task_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def train():
    print("=== Training Task Time Estimation Model ===")
    X, y, encoders, feature_cols = load_task_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.2f} min")
    print(f"  RMSE : {rmse:.2f} min")
    print(f"  R²   : {r2:.4f}")

    cv_scores = cross_val_score(
        xgb.XGBRegressor(n_estimators=100, random_state=42),
        X, y, cv=5, scoring="r2"
    )
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "task_time_model.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "task_time_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "task_time_features.pkl"))
    print("  Saved: task_time_model.pkl\n")

    return {"mae": round(mae, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}


if __name__ == "__main__":
    train()
