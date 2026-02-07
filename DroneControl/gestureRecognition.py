"""Static-gesture feature extraction utilities."""

from __future__ import annotations
from collections.abc import Sequence
from math import sqrt
from typing import Protocol

class _HasXY(Protocol):
    x: float
    y: float


def calculate_distance(reference_x: float, reference_y: float, distant_x: float, distant_y: float) -> float:
    """Compute Euclidean distance between two 2D points."""

    return sqrt((reference_x - distant_x) ** 2 + (reference_y - distant_y) ** 2)


def landmark_normalization(hand_points: Sequence[_HasXY]) -> list[float]:
    """Compute 15 distance-based features from 21 hand landmarks.

    The feature vector is:
    - 5 distances from wrist (landmark 0) to each fingertip (4, 8, 12, 16, 20)
    - 10 pairwise distances between fingertips
    """

    wrist_x = float(hand_points[0].x)
    wrist_y = float(hand_points[0].y)

    fingertip_indices = (4, 8, 12, 16, 20)

    distances: list[float] = []

    for idx in fingertip_indices:
        tip_x = float(hand_points[idx].x)
        tip_y = float(hand_points[idx].y)
        distances.append(calculate_distance(wrist_x, wrist_y, tip_x, tip_y))

    for i in range(len(fingertip_indices)):
        for j in range(i + 1, len(fingertip_indices)):
            idx1 = fingertip_indices[i]
            idx2 = fingertip_indices[j]
            p1_x = float(hand_points[idx1].x)
            p1_y = float(hand_points[idx1].y)
            p2_x = float(hand_points[idx2].x)
            p2_y = float(hand_points[idx2].y)
            distances.append(calculate_distance(p1_x, p1_y, p2_x, p2_y))

    return distances

__all__ = [
    "calculate_distance",
    "landmark_normalization",
    "calculateDistance",
    "LandmarkNormalization",
]