"""Drone control configuration constants."""

from __future__ import annotations
from typing import Final

# --- Prediction thresholds / timing ---
PRED_THRESHOLD_STATIC: Final[float] = 0.95
PRED_THRESHOLD_DYNAMIC: Final[float] = 0.95
COOLDOWN_SECONDS: Final[int] = 2

# --- Drone velocity scaling factors ---
VELOCITY_FACTOR_X: Final[float] = 0.3
VELOCITY_FACTOR_Y: Final[float] = 0.3
VELOCITY_FACTOR_Z: Final[float] = 0.3

# --- Model artifact paths (project-relative) ---
STATIC_MODEL_PATH: Final[str] = "Static/models/140k.joblib"
DYNAMIC_MODEL_PATH: Final[str] = "Dynamic/models/actionNoFlip.keras"
LANDMARKER_MODEL_PATH: Final[str] = "Dynamic/models/hand_landmarker.task"
