"""Run this once to train all 4 models and save them to saved_models/"""
from train_task_time import train as train_task
from train_anomaly import train as train_anomaly
from train_safety_alert import train as train_safety
from train_event_type import train as train_event

if __name__ == "__main__":
    train_task()
    train_anomaly()
    train_safety()
    train_event()
    print("All models trained and saved.")
