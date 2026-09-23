# CAT Operator Guardian — Project Report
### Smart Operator Assistant for CAT Machinery
**Elimination Round Submission**

---

## 1. Project Overview

| Field | Details |
|---|---|
| **Project Title** | Smart Operator Assistant for CAT Machinery |
| **Theme** | Industrial Safety + AI/ML |
| **Goal** | End-to-end intelligent application that supports CAT machinery operators throughout their workday — improving efficiency, safety, and training using smart technologies |

---

## 2. Team Structure

| Role | Responsibility |
|---|---|
| **Game Development** | Simulation module for Operator Training Hub |
| **Frontend** | Daily task dashboard, safety UI, operator interface |
| **Synthetic Data** | Generate labeled training data for ML models |
| **Backend + AI/ML** | ML model training, FastAPI backend, inference API *(this report)* |

---

## 3. Problem Statement

Construction equipment operators (excavators, dozers, loaders, graders) lack intelligent tooling. The system addresses:

- No real-time safety monitoring on the machine
- No prediction of how long tasks will take
- No automated detection of unsafe or unusual machine behavior
- No data-driven training recommendations for operators

---

## 4. Expected Outcomes (from Challenge Brief)

| Feature | Status |
|---|---|
| Daily Task Dashboard | Backend API built |
| Safety Features (seatbelt, proximity, incidents) | ML model + API built |
| Operator Training Hub | Architecture defined (game dev teammate handles simulation) |
| Identify Unusual Behavior | ML model built (XGBoost Anomaly Classifier) |
| Task Time Estimation | ML model built (XGBoost Regressor) |

---

## 5. Dataset Analysis

### 5.1 Files Used

| File | Rows | Purpose |
|---|---|---|
| `telemetry_data.csv` | 5,000 | Real-time machine sensor readings |
| `task_eta_data.csv` | 5,000 | Historical task completion records |
| `safety_events.csv` | 654 | Safety event log with event types |
| `operator_profiles.csv` | 60 | Operator skill and performance scores |

### 5.2 Telemetry Data — Key Columns

| Column | Description |
|---|---|
| timestamp | 2-minute interval sensor snapshots |
| idle_pct | % of time engine was idling |
| speed_kmh, rpm | Machine motion metrics |
| engine_temp_c | Engine temperature |
| fuel_level_pct | Fuel remaining |
| worker_distance_m | Proximity to nearby workers |
| seatbelt_status | Fastened / Unfastened |
| safety_alert | Whether alert was triggered (TRUE/FALSE) |
| anomaly_label | 1 = Normal (4,416 rows), -1 = Anomaly (584 rows) |

### 5.3 Task Data — Key Columns

| Column | Description |
|---|---|
| task_type | Earth Excavation, Trenching, Demolition, etc. |
| weather | Sunny, Rainy, Cloudy, Foggy, Windy |
| soil_condition | Dry, Wet, Loose, Clay, Rocky |
| operator_skill_level | Beginner / Intermediate / Expert |
| estimated_time_min | Planned task duration |
| actual_time_min | Real completion time (target variable) |

### 5.4 Safety Events — Event Type Distribution

| Event Type | Count | % |
|---|---|---|
| normal | 462 | 70.6% |
| proximity_breach | 98 | 15.0% |
| temp_spike | 51 | 7.8% |
| erratic_accel | 21 | 3.2% |
| high_fuel | 12 | 1.8% |
| overspeed | 10 | 1.5% |

### 5.5 Anomaly Distribution in Telemetry

```
Normal  (label = 1 → remapped 0) : 4,416 rows  (88.3%)
Anomaly (label = -1 → remapped 1):   584 rows  (11.7%)
```

---

## 6. AI/ML Models

### Model 1 — Task Time Estimation

| Property | Detail |
|---|---|
| **Algorithm** | XGBoost Regressor |
| **Data** | task_eta_data.csv (5,000 rows) + operator_profiles.csv (join) |
| **Target** | `actual_time_min` (continuous) |
| **Key Features** | task_type, weather, soil_condition, operator_skill_level, operator_experience_years, engine_hours, payload_tonnes, distance_m, idle_time_pct, previous_task_time_min, estimated_time_min, safety_score, fuel_efficiency_score |
| **Evaluation Metrics** | MAE, RMSE, R² + 5-fold cross-validation |
| **Why XGBoost** | Best on tabular structured data, handles mixed types, outperforms neural networks with <10k rows |

---

### Model 2 — Anomaly / Unusual Behavior Detection

| Property | Detail |
|---|---|
| **Algorithm** | XGBoost Classifier (Supervised) |
| **Data** | telemetry_data.csv (5,000 rows) |
| **Target** | `anomaly_label` (0 = Normal, 1 = Anomaly) |
| **Key Features** | idle_pct, speed_kmh, rpm, engine_temp_c, fuel_level_pct, working_fuel_rate_lph, speed_variance, acceleration_events, worker_distance_m, shift_elapsed_min |
| **Class Imbalance Handling** | scale_pos_weight = 4416 / 584 = **7.56** |
| **Evaluation Metrics** | Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| **Why supervised (not Isolation Forest)** | Data is already labeled — supervised classifier gives far better accuracy |

---

### Model 3 — Safety Alert Classifier

| Property | Detail |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Data** | telemetry_data.csv (5,000 rows) |
| **Target** | `safety_alert` (0 = No Alert, 1 = Alert) |
| **Key Features** | seatbelt_status, worker_distance_m, speed_kmh, acceleration_events, idle_pct, risk_level, engine_temp_c, speed_variance |
| **Class Imbalance Handling** | class_weight = 'balanced' |
| **Evaluation Metrics** | Precision, Recall, F1 — **Recall prioritized** (missing a real alert = dangerous) |
| **Why Random Forest** | Handles imbalance natively, provides feature importance (explainability required for safety) |

---

### Model 4 (Bonus) — Event Type Classifier

| Property | Detail |
|---|---|
| **Algorithm** | Random Forest Classifier (Multi-class) |
| **Data** | safety_events.csv (654 rows) |
| **Target** | `event_type` — 6 classes: normal, proximity_breach, erratic_accel, temp_spike, high_fuel, overspeed |
| **Imbalance Handling** | class_weight='balanced' + SMOTE oversampling |
| **Evaluation Metrics** | Per-class Precision, Recall, F1 |
| **Use case** | Classify what type of unsafe behavior is happening in real-time |

---

## 7. Backend Architecture

### Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI (Python) |
| API Documentation | Swagger UI (auto-generated at `/docs`) |
| Database | SQLite + SQLAlchemy ORM |
| ML Serving | joblib model loading at startup |
| Data Validation | Pydantic v2 |

### Database Tables

| Table | Purpose |
|---|---|
| `tasks` | Daily task assignments, status, predicted vs actual time |
| `telemetry_logs` | Incoming sensor readings with auto-predictions |
| `safety_incidents` | Logged safety events and resolution status |
| `operator_profiles` | Seeded from operator_profiles.csv |

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| **ML Predictions** | | |
| POST | `/predict/task-time` | Predict task completion time |
| POST | `/predict/anomaly` | Detect unusual machine behavior |
| POST | `/predict/safety-alert` | Predict if safety alert will trigger |
| POST | `/predict/event-type` | Classify event type |
| **Task Management** | | |
| GET | `/tasks` | Get all tasks (filter by operator/status) |
| POST | `/tasks` | Create task (auto-runs time prediction) |
| PUT | `/tasks/{id}` | Update task status or actual time |
| DELETE | `/tasks/{id}` | Delete task |
| **Telemetry** | | |
| POST | `/telemetry` | Ingest reading (auto anomaly + safety prediction) |
| GET | `/telemetry` | Get logs filtered by machine/operator |
| GET | `/telemetry/anomalies` | Get all detected anomalies |
| **Safety** | | |
| GET | `/incidents` | Get incidents (filter by risk, resolved, operator) |
| POST | `/incidents` | Log new incident |
| PATCH | `/incidents/{id}/resolve` | Mark incident resolved |
| **Operators** | | |
| GET | `/operators` | Get all operator profiles |
| GET | `/operators/{id}` | Get single operator |
| **Dashboard** | | |
| GET | `/dashboard/summary` | KPIs — tasks, alerts, anomalies, incidents count |

---

## 8. Project File Structure

```
CAT-operator-guardian/
├── requirements.txt
├── data/
│   ├── telemetry_data.csv
│   ├── task_eta_data.csv
│   ├── safety_events.csv
│   └── operator_profiles.csv
├── ml/
│   ├── preprocessor.py           ← Loads + cleans all 4 CSVs
│   ├── train_task_time.py        ← Model 1: XGBoost Regressor
│   ├── train_anomaly.py          ← Model 2: XGBoost Classifier
│   ├── train_safety_alert.py     ← Model 3: Random Forest
│   ├── train_event_type.py       ← Model 4: RF Multi-class + SMOTE
│   ├── train_all.py              ← Run once to train all models
│   ├── predict.py                ← Inference functions (used by backend)
│   └── saved_models/             ← Trained .pkl files saved here
└── backend/
    ├── main.py                   ← FastAPI app entry + Swagger + DB init
    ├── config.py                 ← App settings
    ├── database.py               ← SQLAlchemy + SQLite
    ├── db_models.py              ← ORM table definitions
    ├── schemas.py                ← Pydantic request/response schemas
    ├── ml_service.py             ← Bridge between API and ML
    └── routers/
        ├── predict.py            ← /predict/* routes
        ├── tasks.py              ← /tasks/* routes
        ├── telemetry.py          ← /telemetry/* routes
        ├── incidents.py          ← /incidents/* routes
        ├── operators.py          ← /operators/* routes
        └── dashboard.py          ← /dashboard/summary route
```

---

## 9. Data Preprocessing Summary

| Dataset | Transformations Applied |
|---|---|
| telemetry_data | anomaly_label remapped {1→0, -1→1}; safety_alert "TRUE/FALSE" → 1/0; seatbelt Fastened/Unfastened → 0/1; risk_level LOW/MED/HIGH → 0/1/2 |
| task_eta_data | operator_skill_level ordinal encoded; shift_type day/night → 0/1; task_type, weather, soil_condition, machine_type label-encoded; joined with operator_profiles on operator_id |
| safety_events | event_type label-encoded (6 classes); SMOTE applied to minority classes for Model 4 |
| operator_profiles | Used as join table — scores merged into task features |

---

## 10. Team Integration Points

| Teammate | Integration |
|---|---|
| **Frontend** | Consumes all REST API endpoints; Swagger UI provides live documentation |
| **Game Dev** | Simulation sessions POST to `/telemetry` → auto anomaly + safety prediction |
| **Synthetic Data** | CSV schemas defined and locked — teammate generates data matching telemetry + task schemas |

---

## 11. How to Run

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Train all models
cd ml
python train_all.py

# Step 3 — Start backend
cd ../backend
uvicorn main:app --reload --port 8000

# Step 4 — Open Swagger UI
# http://localhost:8000/docs
```

---

## 12. What's Next (Next Round)

- [ ] Complete model training and report accuracy metrics
- [ ] Connect frontend to live API endpoints
- [ ] Integrate game simulation output with `/telemetry` endpoint
- [ ] Add WebSocket support for real-time safety alerts push
- [ ] Deploy backend to cloud (Render / Railway)
- [ ] Add operator training recommendation endpoint based on `weakest_skill`

---

*Report generated: September 2026 | Backend + AI/ML — CAT Operator Guardian Team*
