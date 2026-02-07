"""Gesture label maps used by static and dynamic gesture recognition models."""
from __future__ import annotations
from typing import Final
import numpy as np

# --- Static model output labels ---
STATIC_HAND_GESTURES: Final[dict[int, str]] = {
    0: "BACKWARDS",
    1: "FORWARD",
    2: "HALT",
    3: "TAKEOFF",
    4: "XYCONTROL",
}

# --- Dynamic model action labels ---
# Kept as a NumPy array because downstream code relies on `.shape[0]`.
DYNAMIC_ACTIONS: Final[np.ndarray] = np.array(
    [
        "TAKE_ONOFF",
        "FLY_LEFT",
        "FLY_RIGHT",
        "FLY_UP",
        "FLY_DOWN",
        "FLY_BACK",
        "FLY_FORWARD",
        "ROTATE",
        "FLIP",
        "IDLE",
    ],
    dtype=str,
)
