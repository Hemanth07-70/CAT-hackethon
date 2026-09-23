from fastapi import APIRouter

router = APIRouter(prefix="/metrics", tags=["Model Metrics"])

MODEL_METRICS = {
    "task_time_estimation": {
        "algorithm": "XGBoost Regressor",
        "data_size": 5000,
        "target": "actual_time_min",
        "results": {
            "MAE_min": 7.49,
            "RMSE_min": 10.15,
            "R2_score": 0.9197,
            "CV_R2_mean": 0.9132,
            "CV_R2_std": 0.0077,
        },
        "verdict": "EXCELLENT — Model explains 91.9% of variance in task time",
    },
    "anomaly_detection": {
        "algorithm": "XGBoost Classifier",
        "data_size": 5000,
        "class_balance": {"Normal": 4416, "Anomaly": 584},
        "target": "anomaly_label",
        "results": {
            "accuracy": 0.99,
            "anomaly_precision": 0.94,
            "anomaly_recall": 1.00,
            "anomaly_f1": 0.97,
            "ROC_AUC": 0.9999,
            "confusion_matrix": {
                "true_normal": 875, "false_alert": 8,
                "missed_anomaly": 0, "true_anomaly": 117,
            },
        },
        "verdict": "EXCELLENT — Zero missed anomalies (Recall=1.0), ROC-AUC=0.9999",
    },
    "safety_alert": {
        "algorithm": "Random Forest Classifier",
        "data_size": 5000,
        "class_balance": {"No Alert": 4346, "Alert": 654},
        "target": "safety_alert",
        "results": {
            "accuracy": 1.00,
            "alert_precision": 0.99,
            "alert_recall": 1.00,
            "alert_f1": 1.00,
            "ROC_AUC": 1.0000,
            "confusion_matrix": {
                "true_safe": 868, "false_alert": 1,
                "missed_alert": 0, "true_alert": 131,
            },
        },
        "top_features": {
            "seatbelt_status": 0.636,
            "risk_level": 0.143,
            "worker_distance_m": 0.098,
            "engine_temp_c": 0.066,
            "speed_kmh": 0.025,
        },
        "verdict": "EXCELLENT — Zero missed safety alerts, seatbelt is top predictor",
    },
    "event_type_classifier": {
        "algorithm": "Random Forest + SMOTE",
        "data_size": 654,
        "target": "event_type (6 classes)",
        "results": {
            "overall_accuracy": 0.70,
            "weighted_f1": 0.74,
            "per_class": {
                "normal": {"precision": 0.86, "recall": 0.71, "f1": 0.78},
                "proximity_breach": {"precision": 0.64, "recall": 0.70, "f1": 0.67},
                "temp_spike": {"precision": 1.00, "recall": 1.00, "f1": 1.00},
                "erratic_accel": {"precision": 0.10, "recall": 0.25, "f1": 0.14},
                "high_fuel": {"precision": 0.00, "recall": 0.00, "f1": 0.00},
                "overspeed": {"precision": 1.00, "recall": 0.50, "f1": 0.67},
            },
        },
        "note": "erratic_accel (21 samples) and high_fuel (12 samples) need more synthetic data to improve",
        "verdict": "ACCEPTABLE — temp_spike perfect; minority classes need more training data",
    },
}


@router.get(
    "/",
    summary="Get accuracy metrics for all ML models",
    description="Returns training results, accuracy scores, and evaluation metrics for all 4 models."
)
def get_all_metrics():
    return MODEL_METRICS


@router.get(
    "/{model_name}",
    summary="Get metrics for a specific model",
    description="model_name: task_time_estimation | anomaly_detection | safety_alert | event_type_classifier"
)
def get_model_metrics(model_name: str):
    if model_name not in MODEL_METRICS:
        return {"error": f"Unknown model. Choose from: {list(MODEL_METRICS.keys())}"}
    return MODEL_METRICS[model_name]
