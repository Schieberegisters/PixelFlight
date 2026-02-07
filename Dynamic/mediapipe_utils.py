from typing import Any, Tuple, Final

import cv2
import numpy as np
import mediapipe as mp

# --- MEDIAPIPE SETUP ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- VISUALIZATION CONSTANTS ---
# Colors are BGR
COLOR_FACE_POINT: Final[Tuple[int, int, int]] = (80, 110, 10)
COLOR_FACE_LINE: Final[Tuple[int, int, int]] = (80, 256, 121)

COLOR_POSE_POINT: Final[Tuple[int, int, int]] = (80, 22, 10)
COLOR_POSE_LINE: Final[Tuple[int, int, int]] = (80, 44, 121)

COLOR_LEFT_HAND_POINT: Final[Tuple[int, int, int]] = (121, 22, 76)
COLOR_LEFT_HAND_LINE: Final[Tuple[int, int, int]] = (121, 44, 250)

COLOR_RIGHT_HAND_POINT: Final[Tuple[int, int, int]] = (245, 117, 66)
COLOR_RIGHT_HAND_LINE: Final[Tuple[int, int, int]] = (245, 66, 230)


def mediapipe_detection(image: np.ndarray, model: Any) -> Tuple[np.ndarray, Any]:
    """
    Processes an image through the MediaPipe Holistic model.
    Handles BGR to RGB conversion required by MediaPipe.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Writeable=False improves performance by passing by reference
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image: np.ndarray, results: Any) -> None:
    """Applies custom-styled landmarks to the provided image."""
    
    # Helper for cleaner calls
    def _draw(landmarks, connections, color_point, color_line):
        mp_drawing.draw_landmarks(
            image,
            landmarks,
            connections,
            mp_drawing.DrawingSpec(color=color_point, thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=color_line, thickness=1, circle_radius=1)
        )

    # Face (Contours)
    _draw(results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, COLOR_FACE_POINT, COLOR_FACE_LINE)

    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_POSE_POINT, thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=COLOR_POSE_LINE, thickness=2, circle_radius=2)
    )

    # Left Hand
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_LEFT_HAND_POINT, thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=COLOR_LEFT_HAND_LINE, thickness=2, circle_radius=2)
    )

    # Right Hand
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=COLOR_RIGHT_HAND_POINT, thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=COLOR_RIGHT_HAND_LINE, thickness=2, circle_radius=2)
    )


def annotate_frame(image: np.ndarray, model: Any, styled: bool = True) -> Tuple[np.ndarray, Any]:
    """
    High-level wrapper to run detection and visualize results in one step.
    
    Args:
        image: BGR input frame.
        model: Initialized MediaPipe Holistic instance.
        styled: If True, uses custom colors; otherwise uses default green/red.
    """
    image, results = mediapipe_detection(image, model)

    if styled:
        draw_styled_landmarks(image, results)
    else:
        # Fallback to basic drawing defaults
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    return image, results