"""
Augments safety_events.csv with:
1. New columns: fuel_rate_lph, acceleration_events (added to all existing rows)
2. Synthetic rows for minority classes to reach balanced counts
3. Saves augmented CSV back to data/
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
np.random.seed(42)

# ── per-event distribution parameters ────────────────────────────────────────
DIST = {
    #           speed_kmh      worker_dist   engine_temp   fuel_rate     accel_events  risk_level         seatbelt
    "normal":           dict(spd=(2,9),   dst=(8,30),  tmp=(75,95),  fuel=(5,10),  acc=(0,3),  risk=["LOW"]*8+["MEDIUM"]*2,        belt=["Fastened"]*5+["Unfastened"]*5),
    "proximity_breach": dict(spd=(3,10),  dst=(0.3,3), tmp=(78,95),  fuel=(6,13),  acc=(1,5),  risk=["HIGH"]*7+["MEDIUM"]*3,       belt=["Fastened"]*8+["Unfastened"]*2),
    "temp_spike":       dict(spd=(2,7),   dst=(8,25),  tmp=(100,130),fuel=(10,18), acc=(0,4),  risk=["MEDIUM"]*5+["HIGH"]*5,       belt=["Fastened"]*6+["Unfastened"]*4),
    "erratic_accel":    dict(spd=(4,16),  dst=(5,22),  tmp=(80,102), fuel=(13,22), acc=(6,15), risk=["MEDIUM"]*5+["HIGH"]*5,       belt=["Fastened"]*4+["Unfastened"]*6),
    "high_fuel":        dict(spd=(3,8),   dst=(10,28), tmp=(85,105), fuel=(20,35), acc=(2,6),  risk=["MEDIUM"]*7+["LOW"]*3,        belt=["Fastened"]*6+["Unfastened"]*4),
    "overspeed":        dict(spd=(12,22), dst=(5,22),  tmp=(85,105), fuel=(14,22), acc=(3,9),  risk=["HIGH"]*6+["MEDIUM"]*4,       belt=["Fastened"]*5+["Unfastened"]*5),
}

WEATHERS  = ["Sunny","Rainy","Cloudy","Windy","Foggy"]
MACHINES  = ["Dozer","Excavator","Backhoe","Loader","Grader"]
MACHINE_IDS = {
    "Dozer": ["DZR001","DZR002","DZR003","DZR004"],
    "Excavator": ["EXC001","EXC002","EXC003","EXC004","EXC005"],
    "Backhoe": ["BCK001","BCK002","BCK003","BCK004"],
    "Loader": ["LDR001","LDR002","LDR003"],
    "Grader": ["GRD001","GRD002","GRD003"],
}
OPERATORS = [f"OP{str(i).zfill(4)}" for i in range(1001, 1061)]
BASE_TS   = datetime(2025, 3, 10, 6, 0, 0)


def sample_row(event_type: str, idx: int) -> dict:
    d = DIST[event_type]
    machine_type = np.random.choice(MACHINES)
    machine_id   = np.random.choice(MACHINE_IDS[machine_type])
    ts = BASE_TS + timedelta(hours=idx * 0.25 + np.random.uniform(0, 0.2))

    return {
        "event_id":          f"SYNTH{str(idx).zfill(6)}",
        "timestamp":         ts.strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id":        machine_id,
        "machine_type":      machine_type,
        "operator_id":       np.random.choice(OPERATORS),
        "shift_type":        "day" if 6 <= ts.hour < 20 else "night",
        "event_type":        event_type,
        "speed_kmh":         round(np.random.uniform(*d["spd"]), 2),
        "worker_distance_m": round(np.random.uniform(*d["dst"]), 1),
        "engine_temp_c":     round(np.random.uniform(*d["tmp"]), 1),
        "weather":           np.random.choice(WEATHERS),
        "seatbelt_status":   np.random.choice(d["belt"]),
        "risk_level":        np.random.choice(d["risk"]),
        "response_time_sec": round(np.random.uniform(1, 15), 1),
        "resolved":          np.random.choice([True, False], p=[0.75, 0.25]),
        "fuel_rate_lph":     round(np.random.uniform(*d["fuel"]), 2),
        "acceleration_events": int(np.random.randint(*d["acc"])),
    }


def add_new_cols_to_existing(df: pd.DataFrame) -> pd.DataFrame:
    """Add fuel_rate_lph and acceleration_events to every existing row."""
    fuels, accels = [], []
    for _, row in df.iterrows():
        d = DIST.get(row["event_type"], DIST["normal"])
        fuels.append(round(np.random.uniform(*d["fuel"]), 2))
        accels.append(int(np.random.randint(*d["acc"])))
    df["fuel_rate_lph"]       = fuels
    df["acceleration_events"] = accels
    return df


def augment():
    df = pd.read_csv(os.path.join(DATA_DIR, "safety_events.csv"))
    print(f"Original rows: {len(df)}")

    # Step 1 — add new columns to existing rows
    df = add_new_cols_to_existing(df)

    # Step 2 — generate synthetic rows for minority classes
    TARGETS = {
        "erratic_accel":   150,
        "high_fuel":       150,
        "overspeed":        80,
        "temp_spike":      120,
        "proximity_breach":160,
        "normal":          462,   # keep as is
    }

    new_rows = []
    idx = 10000
    for event_type, target in TARGETS.items():
        current = len(df[df["event_type"] == event_type])
        needed  = max(0, target - current)
        print(f"  {event_type:20s}: existing={current:3d}  generating={needed:3d}  total={current+needed}")
        for _ in range(needed):
            new_rows.append(sample_row(event_type, idx))
            idx += 1

    synthetic_df = pd.DataFrame(new_rows)
    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
    augmented_df  = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "safety_events_augmented.csv")
    augmented_df.to_csv(out_path, index=False)

    print(f"\nAugmented total rows : {len(augmented_df)}")
    print("Class distribution after augmentation:")
    print(augmented_df["event_type"].value_counts().to_string())
    print(f"\nSaved → {out_path}")
    return out_path


if __name__ == "__main__":
    augment()
