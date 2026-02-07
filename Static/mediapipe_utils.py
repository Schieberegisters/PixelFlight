"""
MediaPipe HandLandmarker helpers used by the static model training scripts.
Targeting MediaPipe Tasks API with full static type support.
"""

from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Any, Final, Literal, Optional, Tuple, List

# Direct imports to provide real types to Pylance and the IDE
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

# --- CONFIGURATION ---
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODELS_DIR: Final[Path] = BASE_DIR/ "models"

# --- TYPES ---
RunningModeStr = Literal["VIDEO", "IMAGE"]


def get_hand_landmarker(
    model_path: str | Path | None = None,
    running_mode: RunningModeStr = "VIDEO",
    num_hands: int = 1,
    min_hand_detection_confidence: float = 0.3,
    min_hand_presence_confidence: float = 0.3,
    min_tracking_confidence: float = 0.3,
) -> HandLandmarker:
    """Initializes and returns a configured MediaPipe HandLandmarker instance."""

    if model_path is None:
        model_path = MODELS_DIR / "hand_landmarker.task"

    task_path = Path(model_path)
    if not task_path.exists():
        raise FileNotFoundError(f"Model not found: {task_path}")

    mode_map = {
        "VIDEO": RunningMode.VIDEO,
        "IMAGE": RunningMode.IMAGE,
    }

    if running_mode not in mode_map:
        raise ValueError(f"Invalid running_mode={running_mode!r}. Use 'VIDEO' or 'IMAGE'.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_path)),
        running_mode=mode_map[running_mode],
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return HandLandmarker.create_from_options(options)


def process_frame_with_mediapipe(
    frame_bgr: np.ndarray,
    landmarker: HandLandmarker, 
    timestamp_ms: int,
    use_video_mode: bool = True,
) -> Tuple[Optional[Any], mp.Image]:
    """Runs HandLandmarker inference on a BGR frame."""
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)

    try:
        if use_video_mode:
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = landmarker.detect(mp_image)
    except Exception as exc:
        print(f"MediaPipe error: {exc}")
        return None, mp_image

    return result, mp_image


def draw_hand_landmarks(
    frame_bgr: np.ndarray, 
    hand_landmarks: List[Any], 
    connections: Optional[Any] = None
) -> np.ndarray:
    """Renders hand landmarks and connections onto the frame in-place."""

    if connections is None:
        connections = mp.solutions.hands.HAND_CONNECTIONS

    h, w = frame_bgr.shape[:2]
    COLOR_LINK = (0, 255, 0)
    COLOR_NODE = (255, 0, 0)

    for landmarks in hand_landmarks:
        for connection in connections:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            
            p1 = (int(start.x * w), int(start.y * h))
            p2 = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame_bgr, p1, p2, COLOR_LINK, 2)

        wrist = landmarks[0]
        cv2.circle(frame_bgr, (int(wrist.x * w), int(wrist.y * h)), 5, COLOR_NODE, -1)

    return frame_bgr