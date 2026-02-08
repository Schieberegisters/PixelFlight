from typing import Any, Final
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- DATA LAYOUT CONSTANTS ---
# Pose: 33 landmarks * (x, y, z, visibility)
# Face: 468 landmarks * (x, y, z)
# Hands: 21 landmarks * (x, y, z) per hand
POSE_LEN: Final[int] = 33 * 4
FACE_LEN: Final[int] = 468 * 3
HAND_LEN: Final[int] = 21 * 3
TOTAL_LEN: Final[int] = POSE_LEN + FACE_LEN + (HAND_LEN * 2)


def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extracts landmarks from MediaPipe Holistic results into a flattened 1662-dim vector.
    Missing landmarks are filled with zeros to maintain a consistent vector shape.
    """

    pose = (
        np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks else np.zeros(POSE_LEN)
    )

    face = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks else np.zeros(FACE_LEN)
    )

    lh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks else np.zeros(HAND_LEN)
    )

    rh = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks else np.zeros(HAND_LEN)
    )

    # Concatenate all features into the final feature vector
    keypoints = np.concatenate([pose, face, lh, rh])

    if keypoints.shape != (TOTAL_LEN,):
        raise ValueError(f"Vector shape mismatch: {keypoints.shape} (expected {TOTAL_LEN})")

    return keypoints