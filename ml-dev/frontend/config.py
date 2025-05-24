"""Configuration partagée frontend ↔ backend"""
import os

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://192.168.2.222:8000")
MLFLOW_TRACKING_URL: str = os.getenv("MLFLOW_TRACKING_URL", "http://192.168.2.222:5000")