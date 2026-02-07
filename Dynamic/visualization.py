from __future__ import annotations
from typing import Final, List, Optional, Union

import cv2
import numpy as np

# --- CONFIG IMPORTS ---
from config.dynamic import COLORS

# --- UI CONSTANTS ---
BAR_HEIGHT: Final[int] = 40
BAR_THICKNESS: Final[int] = 30
TEXT_OFFSET_Y: Final[int] = 25
START_Y: Final[int] = 60


def render_probability_bars(
    probabilities: Union[List[float], np.ndarray],
    actions: Union[List[str], np.ndarray],
    frame: np.ndarray,
    bar_colors: Optional[List[tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Renders real-time probability bars and action labels onto the video frame."""
    
    if bar_colors is None:
        bar_colors = COLORS
        
    output_frame = frame.copy()

    for idx, prob in enumerate(probabilities):
        # Calculate bar geometry
        y_top = START_Y + (idx * BAR_HEIGHT)
        y_bottom = y_top + BAR_THICKNESS
        bar_width = int(prob * 100)

        # Draw probability bar
        cv2.rectangle(
            output_frame,
            (0, y_top),
            (bar_width, y_bottom),
            bar_colors[idx % len(bar_colors)],
            -1
        )

        # Draw action label
        cv2.putText(
            output_frame,
            actions[idx],
            (5, y_top + TEXT_OFFSET_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    return output_frame