"""Dynamic gesture configuration constants (dataset, model, and inference settings)."""

from __future__ import annotations
from pathlib import Path
from typing import Final

# --- Paths (project-relative) ---
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BASE_DIR: Final[Path] = PROJECT_ROOT  # Backwards-compatible alias
DYNAMIC_DIR: Final[Path] = PROJECT_ROOT / "Dynamic"

DATASET_DIR: Final[Path] = DYNAMIC_DIR / "TrainingData" / "DynamicRecognition_relational"
DATA_PATH: Final[str] = str(DATASET_DIR)

# --- Dataset / sequence parameters ---
NO_SEQUENCES: Final[int] = 200
SEQUENCE_LENGTH: Final[int] = 20
STABLE_LENGTH: Final[int] = 10
INPUT_FEATURES: Final[int] = 1662
INPUT_SHAPE: Final[tuple[int, int]] = (SEQUENCE_LENGTH, INPUT_FEATURES)

# --- ST-GCN configuration ---
# Pose (33) + left hand (21) + right hand (21)
NUM_NODES: Final[int] = 75
CHANNELS: Final[int] = 3  # x, y, z

# --- Training hyperparameters ---
LEARNING_RATE: Final[float] = 1e-3
TEST_SIZE: Final[float] = 0.20
EPOCHS: Final[int] = 250
OPTIMIZER: Final[str] = "Adam"
LOSS: Final[str] = "categorical_crossentropy"
METRICS: Final[list[str]] = ["categorical_accuracy"]

# --- Model / logs paths ---
MODEL_PATH: Final[str] = str(DYNAMIC_DIR / "models" / "actionNoFlip.keras")
LOG_DIR: Final[str] = str(PROJECT_ROOT / "Logs")

# --- Inference parameters ---
THRESHOLD: Final[float] = 0.8
MAX_SENTENCE_LENGTH: Final[int] = 5

# --- Visualization colors (BGR) ---
COLORS: Final[list[tuple[int, int, int]]] = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (245, 117, 16),  # Dark orange
    (117, 245, 16),  # Lime green
    (16, 117, 245),  # Electric blue
    (128, 0, 128),  # Purple
    (128, 0, 0),  # Maroon
    (0, 128, 128),  # Teal
    (128, 128, 0),  # Olive
    (0, 0, 128),  # Navy
    (255, 192, 203),  # Pink
    (128, 128, 128),  # Gray
]
