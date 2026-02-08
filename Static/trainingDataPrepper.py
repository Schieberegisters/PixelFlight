"""
Webcam data collection script for the static gesture model.
Captures hand landmarks and saves them to CSV for training.
"""

from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from typing import Final, List, Dict, Union
import cv2
import numpy as np
from Static.data_utils import (
    flip_frame, 
    normalize_landmarks, 
    save_data_to_csv, 
    expand_training_data
)
from Static.mediapipe_utils import (
    get_hand_landmarker, 
    process_frame_with_mediapipe, 
    draw_hand_landmarks
)

# --- CONFIGURATION ---
BASE_DIR: Final[Path] = Path(__file__).resolve().parent
OUTPUT_CSV_FILE: Final[Path] = BASE_DIR / "gesture_training_data_teststs.csv"
MODEL_ASSET_PATH: Final[Path] = BASE_DIR / "models" / "hand_landmarker.task"

# Mapping keys to Class Names
GESTURE_MAP: Final[Dict[int, str]] = {
    ord('1'): 'FORWARD',
    ord('2'): 'BACKWARDS',
    ord('3'): 'TAKEOFF',
    ord('4'): 'LANDING',
    ord('5'): 'XYCONTROL'
}


class DataCollector:
    """Manages the video loop, gesture state, and data buffering."""

    def __init__(self, camera_index: int = 0):
        self.camera = cv2.VideoCapture(camera_index)
        
        # Data State
        self.data_buffer: List[List[Union[float, str]]] = []
        self.current_gesture: str = "NONE"
        self.is_collecting: bool = False
        
        # Initialize MediaPipe (Video Mode)
        self.landmarker = get_hand_landmarker(
            model_path=MODEL_ASSET_PATH,
            running_mode="VIDEO",
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _extract_features(self, frame: np.ndarray, timestamp_ms: int) -> tuple[list[float], bool, np.ndarray]:
        """
        Process frame: Detect -> Draw -> Normalize features.
        Returns: (features_list, hand_detected_bool, annotated_frame).
        """
        result, _ = process_frame_with_mediapipe(frame, self.landmarker, timestamp_ms)
        
        hand_found = False
        features = []

        if result and result.hand_landmarks:
            hand_found = True
            # Use the manual drawing utility
            frame = draw_hand_landmarks(frame, result.hand_landmarks)
            # Normalize the first detected hand
            features = normalize_landmarks(result.hand_landmarks[0])

        return features, hand_found, frame

    def _draw_ui(self, frame: np.ndarray) -> None:
        """Overlays status text onto the frame."""
        status_color = (0, 255, 0) if self.is_collecting else (0, 165, 255) # Green vs Orange
        state_text = "RECORDING" if self.is_collecting else "PAUSED"
        
        info_text = (
            f"Target: {self.current_gesture} | {state_text} | Samples: {len(self.data_buffer)}"
        )

        cv2.putText(
            frame, info_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA
        )
        
        # Helper text
        if not self.is_collecting:
            cv2.putText(
                frame, "Keys: [1-5] Select Class | [S] Start | [F] Stop | [Q] Quit", 
                (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
            )

    def _cleanup(self) -> None:
        """Release resources and trigger save dialog."""
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
        self._save_dialog()

    def _save_dialog(self) -> None:
        """Handles the post-session saving and augmentation workflow."""
        print(f"\nSession ended. Collected {len(self.data_buffer)} samples.")
        
        if not self.data_buffer:
            print("No data to save.")
            return

        save = input(f"Save raw data to {OUTPUT_CSV_FILE.name}? [y/n]: ").strip().lower()
        if save == 'y':
            save_data_to_csv(self.data_buffer, str(OUTPUT_CSV_FILE), mode="a")
            print(f"Raw data saved.")

            expand = input("Auto-augment data (scale variations)? [y/n]: ").strip().lower()
            if expand == 'y':
                expand_training_data(str(OUTPUT_CSV_FILE))
            else:
                print("Skipping augmentation.")
        else:
            print("Data discarded.")


def main():
    app = DataCollector()
    
    try:
        while app.camera.isOpened():
            ret, frame = app.camera.read()
            if not ret: 
                print("Error reading frame.")
                break

            # Pre-processing
            frame = flip_frame(frame)
            ts = int(app.camera.get(cv2.CAP_PROP_POS_MSEC))

            # Inference
            features, hand_found, frame = app._extract_features(frame, ts)

            # Record Data
            if app.is_collecting and hand_found and features:
                app.data_buffer.append(list(features) + [app.current_gesture])

            # Visualization
            app._draw_ui(frame)
            cv2.imshow("Gesture Data Collector", frame)

            # Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key in GESTURE_MAP:
                app.current_gesture = GESTURE_MAP[key]
                app.is_collecting = False
                print(f"[UI] Target set: {app.current_gesture}")
            elif key == ord('s'):
                if app.current_gesture != "NONE":
                    app.is_collecting = True
                    print("[UI] >>> RECORDING STARTED")
                else:
                    print("[UI] !! Select a class first")
            elif key == ord('f'):
                app.is_collecting = False
                print("[UI] || PAUSED")
            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        app._cleanup()


if __name__ == "__main__":
    main()