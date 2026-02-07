import threading
import warnings
from collections import deque
from time import time
from typing import Optional, List, Tuple, Any, Deque

import cv2
import joblib
import numpy as np
import mediapipe as mp
from djitellopy import Tello

# --- CONFIG IMPORTS ---
from config.drone import (
    PRED_THRESHOLD_STATIC,
    COOLDOWN_SECONDS,
    VELOCITY_FACTOR_X,
    VELOCITY_FACTOR_Y,
    VELOCITY_FACTOR_Z,
    STATIC_MODEL_PATH,
    DYNAMIC_MODEL_PATH,
    LANDMARKER_MODEL_PATH
)
from config.gestures import STATIC_HAND_GESTURES, DYNAMIC_ACTIONS
from config.dynamic import SEQUENCE_LENGTH, STABLE_LENGTH
from config.mpParameters import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

# --- LOCAL MODULES ---
from .States import CurrentCommand
from .gestureRecognition import landmark_normalization

# Dynamic model utilities
from Dynamic.keypoints import extract_keypoints
from Dynamic.mediapipe_utils import annotate_frame, mp_holistic
from Dynamic.model import load_model as load_dynamic_model


class DroneControl:
    """
    Handles drone control logic via gesture recognition (static and dynamic).
    """

    # Action Groupings
    SPECIAL_ACTIONS: List[str] = ["TAKE_ONOFF", "FLIP"]
    MOVEMENT_ACTIONS: List[str] = [
        "FLY_FORWARD", "FLY_BACK",
        "FLY_LEFT", "FLY_RIGHT",
        "FLY_UP", "FLY_DOWN"
    ]

    def __init__(
        self,
        debug: bool = False,
        static_model_path: str = STATIC_MODEL_PATH,
        dynamic_model_path: Optional[str] = DYNAMIC_MODEL_PATH,
        landmarker_model_path: str = LANDMARKER_MODEL_PATH,
    ) -> None:
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

        self.debug = debug
        
        self.static_model = joblib.load(static_model_path)
        self.dynamic_model = load_dynamic_model(dynamic_model_path)

        self.drone: Optional[Tello] = None
        
        # Command States
        self.command_lr = CurrentCommand.IDLE
        self.command_ud = CurrentCommand.IDLE
        self.command_fb = CurrentCommand.IDLE
        self.landing = CurrentCommand.IDLE
        self.takeoff = CurrentCommand.IDLE
        self.flip = CurrentCommand.IDLE

        # Threading
        self.special_thread: Optional[threading.Thread] = None
        self.thread_lock = threading.Lock()
        self.last_special_time: float = 0.0

        # Sequence Management
        self.sequence: Deque[List[float]] = deque(maxlen=SEQUENCE_LENGTH)
        self.predictions: List[int] = []
        self.last_dynamic_action: Optional[Tuple[str, float]] = None
        self.last_static_state: Optional[str] = None

        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.camera = cv2.VideoCapture(0)

        if not self.debug:
            self._connect_drone()

    def _connect_drone(self) -> None:
        """Initializes connection to the Tello drone."""
        try:
            self.drone = Tello()
            self.drone.connect()
            print(f"Battery: {self.drone.get_battery()}%")
        except Exception as e:
            print(f"Error connecting to drone: {e}")

    @staticmethod
    def _flip_image(frame: np.ndarray) -> np.ndarray:
        """Horizontally flips the provided image frame."""
        return cv2.flip(frame, 1)

    def _predict_static(self, right_hand_landmarks: Any) -> Optional[str]:
        """Predicts static gesture from hand landmarks."""
        if right_hand_landmarks is None:
            return None

        features = np.array(landmark_normalization(right_hand_landmarks)).reshape(1, -1)
        proba = self.static_model.predict_proba(features)[0]

        if np.max(proba) < PRED_THRESHOLD_STATIC:
            return None

        state_idx = int(np.argmax(proba))
        state = STATIC_HAND_GESTURES.get(state_idx)
        self.last_static_state = state
        return state

    def _predict_dynamic(self, keypoints: List[float]) -> Optional[str]:
        """Predicts dynamic action based on keypoint sequence."""
        self.sequence.append(keypoints)
        if len(self.sequence) < SEQUENCE_LENGTH:
            return None

        seq_array = np.expand_dims(np.array(self.sequence), axis=0)
        preds = self.dynamic_model.predict(seq_array)
        best_idx = int(np.argmax(preds[0]))
        current_prob = float(preds[0][best_idx])

        self.predictions.append(best_idx)
        self.predictions = self.predictions[-STABLE_LENGTH:]

        # Ensure prediction stability over time
        if len(self.predictions) == STABLE_LENGTH and len(np.unique(self.predictions)) == 1:
            valid_action = DYNAMIC_ACTIONS[best_idx]
            self.last_dynamic_action = (valid_action, current_prob)
            return valid_action

        return None

    def _reset_commands(self) -> None:
        """Resets all directional commands to IDLE."""
        self.command_fb = CurrentCommand.IDLE
        self.command_lr = CurrentCommand.IDLE
        self.command_ud = CurrentCommand.IDLE
        self.flip = CurrentCommand.IDLE

    def _handle_gesture_logic(self, static_gesture: Optional[str], dynamic_action: Optional[str]) -> None:
        """Maps recognized gestures to drone commands."""
        self._reset_commands()

        if static_gesture == "HALT":
            return

        if dynamic_action in self.MOVEMENT_ACTIONS:
            # Mapping action strings to command states
            action_map = {
                "FLY_FORWARD": ("command_fb", CurrentCommand.FLY_FORWARD),
                "FLY_BACK": ("command_fb", CurrentCommand.FLY_BACK),
                "FLY_LEFT": ("command_lr", CurrentCommand.FLY_LEFT),
                "FLY_RIGHT": ("command_lr", CurrentCommand.FLY_RIGHT),
                "FLY_UP": ("command_ud", CurrentCommand.FLY_UP),
                "FLY_DOWN": ("command_ud", CurrentCommand.FLY_DOWN),
            }
            if dynamic_action in action_map:
                attr, cmd = action_map[dynamic_action]
                setattr(self, attr, cmd)

        elif dynamic_action in self.SPECIAL_ACTIONS:
            if dynamic_action == "TAKE_ONOFF":
                if self.drone and self.drone.is_flying:
                    self.landing = CurrentCommand.LANDING
                    self.takeoff = CurrentCommand.IDLE
                elif self.drone:
                    self.takeoff = CurrentCommand.TAKE_OFF
                    self.landing = CurrentCommand.IDLE
                elif self.debug:
                    print("DEBUG: Toggle Takeoff/Landing triggered")

            elif dynamic_action == "FLIP":
                self.flip = CurrentCommand.FLIP

    def _draw_status(self, frame: np.ndarray) -> None:
        """Overlays current gesture status on the frame."""
        status_lines = []
        if self.last_static_state:
            status_lines.append(f"Static: {self.last_static_state}")
        if self.last_dynamic_action:
            act, prob = self.last_dynamic_action
            status_lines.append(f"Dynamic: {act} ({prob:.2f})")

        y_pos = 40
        for text in status_lines:
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 25

    def _update_cooldown_overlay(self, frame: np.ndarray) -> None:
        """Displays cooldown timer if a special action was recently executed."""
        if not self.last_special_time:
            return

        elapsed = time() - self.last_special_time
        if elapsed < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - elapsed)
            cv2.putText(
                frame,
                f"Cooldown: {remaining}s",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    def _calculate_motor_power(self) -> Tuple[int, int, int, int]:
        """Calculates RC control values based on current command states."""
        lr = fb = ud = yv = 0
        speed_x = int(100 * VELOCITY_FACTOR_X)
        speed_y = int(100 * VELOCITY_FACTOR_Y)
        speed_z = int(100 * VELOCITY_FACTOR_Z)

        if self.command_lr == CurrentCommand.FLY_LEFT:
            lr = -speed_x
        elif self.command_lr == CurrentCommand.FLY_RIGHT:
            lr = speed_x

        if self.command_ud == CurrentCommand.FLY_UP:
            ud = speed_y
        elif self.command_ud == CurrentCommand.FLY_DOWN:
            ud = -speed_y

        if self.command_fb == CurrentCommand.FLY_FORWARD:
            fb = speed_z
        elif self.command_fb == CurrentCommand.FLY_BACK:
            fb = -speed_z

        return lr, fb, ud, yv

    def _execute_special_action_thread(self) -> None:
        """Worker function for executing blocking special drone commands."""
        if self.drone is None:
            return

        if self.landing == CurrentCommand.LANDING:
            self.drone.land()
            self.landing = CurrentCommand.IDLE

        if self.takeoff == CurrentCommand.TAKE_OFF:
            self.drone.takeoff()
            self.takeoff = CurrentCommand.IDLE
        
        if self.flip == CurrentCommand.FLIP:
            self.drone.flip_forward()
            self.flip = CurrentCommand.IDLE

        with self.thread_lock:
            self.last_special_time = time()

    def _dispatch_drone_commands(self) -> None:
        """Routes logic to either RC control or special action threads."""
        if self.drone is None:
            return

        requires_special = (
            self.landing == CurrentCommand.LANDING or 
            self.takeoff == CurrentCommand.TAKE_OFF or 
            self.flip == CurrentCommand.FLIP
        )
        
        cooldown_active = (time() - self.last_special_time) < COOLDOWN_SECONDS

        if requires_special and not cooldown_active:
            with self.thread_lock:
                if self.special_thread is None or not self.special_thread.is_alive():
                    print("Starting special command")
                    self.special_thread = threading.Thread(
                        target=self._execute_special_action_thread, 
                        daemon=True
                    )
                    self.special_thread.start()
        else:
            # Standard flight control
            self.drone.send_rc_control(*self._calculate_motor_power())

    def _process_frame(self, frame: np.ndarray) -> None:
        """Main processing pipeline: annotation, prediction, and UI drawing."""
        frame, results = annotate_frame(frame, self.holistic, styled=True)
        
        # Extract landmarks safely
        right_hand = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None
        
        # Prepare skeleton view
        black_frame = np.zeros(frame.shape, dtype=np.uint8)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(black_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(black_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(black_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Predictions
        static_gesture = self._predict_static(right_hand)
        keypoints = extract_keypoints(results)
        dynamic_action = self._predict_dynamic(keypoints)

        self._handle_gesture_logic(static_gesture, dynamic_action)
        
        # Visualization
        frame = self._flip_image(frame)
        black_frame = self._flip_image(black_frame)
        
        self._update_cooldown_overlay(black_frame)
        self._draw_status(black_frame)

        cv2.imshow("Landmarker", frame)      
        cv2.imshow("Skeleton Only", black_frame)

    def run(self) -> None:
        """Starts the main application loop."""
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("Error reading frame")
                    break

                self._process_frame(frame)

                if not self.debug:
                    self._dispatch_drone_commands()

                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Releases resources and ensures safe shutdown."""
        if self.special_thread and self.special_thread.is_alive():
            self.special_thread.join(timeout=2.0)
        
        if not self.debug and self.drone:
            self.drone.land()
        
        self.camera.release()
        cv2.destroyAllWindows()