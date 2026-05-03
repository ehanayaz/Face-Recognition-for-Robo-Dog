"""Application data directories (relative to project root)."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
MODEL_DIR = DATA_DIR / "model"
TEMP_ENROLL_DIR = DATA_DIR / "temp_enroll"
SETTINGS_PATH = DATA_DIR / "app_settings.json"


def ensure_data_dirs() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_ENROLL_DIR.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = {
    "k": 50,
    "distance_threshold": None,
    "camera_index": 0,
    "crop_margin": 0.25,
    "square_crop": False,
}


def load_settings() -> dict:
    ensure_data_dirs()
    if not SETTINGS_PATH.is_file():
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)
    out = dict(DEFAULT_SETTINGS)
    out.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
    if "distance_threshold" in data and data["distance_threshold"] is not None:
        out["distance_threshold"] = float(data["distance_threshold"])
    return out


def save_settings(settings: dict) -> None:
    ensure_data_dirs()
    merged = dict(DEFAULT_SETTINGS)
    merged.update(settings)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def effective_threshold(settings: dict, model_suggested: float | None) -> float:
    from face_recognition_pca.config import DEFAULT_DISTANCE_THRESHOLD

    t = settings.get("distance_threshold")
    if t is not None:
        return float(t)
    if model_suggested is not None:
        return float(model_suggested)
    return DEFAULT_DISTANCE_THRESHOLD
