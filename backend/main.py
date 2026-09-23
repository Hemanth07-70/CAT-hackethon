import os
import sys
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import engine, SessionLocal
import db_models
from ml_service import load_all_models
from routers import predict, tasks, telemetry, incidents, operators, dashboard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ml"))


def seed_operators():
    """Load operator profiles CSV into DB on first run."""
    db = SessionLocal()
    try:
        if db.query(db_models.OperatorProfile).count() > 0:
            return
        data_path = os.environ.get(
            "CAT_DATA_DIR",
            "/Users/hemanthchowdary/Downloads/CAT/data"
        )
        csv_path = os.path.join(data_path, "operator_profiles.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: operator_profiles.csv not found at {csv_path}")
            return
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            profile = db_models.OperatorProfile(
                operator_id=row["operator_id"],
                skill_level=row["skill_level"],
                experience_years=float(row["experience_years"]),
                safety_score=int(row["safety_score"]),
                fuel_efficiency_score=int(row["fuel_efficiency_score"]),
                precision_score=int(row["precision_score"]),
                idle_management_score=int(row["idle_management_score"]),
                response_time_score=int(row["response_time_score"]),
                weakest_skill=row["weakest_skill"],
                shifts_completed=int(row["shifts_completed"]),
                avg_idle_pct=float(str(row["avg_idle_pct"]).replace("5", "5")),
            )
            db.add(profile)
        db.commit()
        print(f"Seeded {len(df)} operator profiles.")
    except Exception as e:
        print(f"Operator seeding error: {e}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_models.Base.metadata.create_all(bind=engine)
    seed_operators()
    try:
        load_all_models()
    except Exception as e:
        print(f"Warning: Could not load ML models — {e}")
        print("Run ml/train_all.py first to train the models.")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## CAT Operator Guardian — Smart Operator Assistant API

This API powers an intelligent assistant for CAT (Caterpillar) machinery operators.

### ML Models
| Model | Algorithm | Purpose |
|---|---|---|
| Task Time Estimation | XGBoost Regressor | Predict task completion time |
| Anomaly Detection | XGBoost Classifier | Detect unusual machine behavior |
| Safety Alert | Random Forest | Predict safety alert risk |
| Event Type | Random Forest | Classify event type (proximity_breach, erratic_accel, etc.) |

### Key Features
- **Daily Task Dashboard** — create, track, and update tasks
- **Real-time Telemetry** — ingest sensor data with automatic anomaly + safety prediction
- **Safety Incidents** — log and resolve safety events
- **Operator Profiles** — view operator performance scores
- **Dashboard Summary** — KPIs for the frontend

### Swagger UI available at `/docs` | ReDoc at `/redoc`
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(tasks.router)
app.include_router(telemetry.router)
app.include_router(incidents.router)
app.include_router(operators.router)
app.include_router(dashboard.router)


@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs"
    }
