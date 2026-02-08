import argparse
import math
import os
import sys
import random
import shutil
import time
from typing import Final, List, Tuple, Optional
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- LOCAL MODULES ---
from keypoints import extract_keypoints
from mediapipe_utils import mp_holistic, mediapipe_detection, draw_styled_landmarks
from config.dynamic import  NO_SEQUENCES, SEQUENCE_LENGTH,DATASET_DIR
from config.gestures import DYNAMIC_ACTIONS
from config.mpParameters import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

# --- CONSTANTS ---
ACTIONS: Final[np.ndarray] = DYNAMIC_ACTIONS

# Data Layout
POSE_LEN: Final[int] = 33 * 4
FACE_LEN: Final[int] = 468 * 3
HAND_LEN: Final[int] = 21 * 3

# Anatomical mirroring pairs (Left Index, Right Index)
POSE_SWAP_PAIRS: Final[List[Tuple[int, int]]] = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), 
    (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), 
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]


class DataAugmenter:
    """Handles geometric transformations for data augmentation."""

    @staticmethod
    def get_centroid(keypoints: np.ndarray) -> Tuple[float, float]:
        """Calculates the centroid (x, y) of the pose, ignoring zero-values."""
        # Extract X and Y coordinates from flattened pose array
        # Pose data is [x1, y1, z1, v1, x2, y2, ...]
        pose_data = keypoints[:POSE_LEN]
        xs = pose_data[0::4]
        ys = pose_data[1::4]

        # Filter out zero values (undetected landmarks)
        valid_mask = xs != 0
        if not np.any(valid_mask):
            return 0.5, 0.5

        return float(np.mean(xs[valid_mask])), float(np.mean(ys[valid_mask]))

    @staticmethod
    def flip_keypoints(keypoints: np.ndarray) -> np.ndarray:
        """Horizontally flips keypoints and swaps anatomical left/right sides."""
        flipped = keypoints.copy()

        # 1. Flip all X coordinates (index 0, 3, 6...)
        # We process the entire array, but we need to handle the stride differences
        # Pose (stride 4), Face/Hands (stride 3)
        
        # Pose X-flip
        for i in range(0, POSE_LEN, 4):
            if flipped[i] != 0: 
                flipped[i] = 1.0 - flipped[i]

        # Face + Hands X-flip
        for i in range(POSE_LEN, len(flipped), 3):
            if flipped[i] != 0: 
                flipped[i] = 1.0 - flipped[i]

        # 2. Swap Anatomical Sides for Pose
        for left_idx, right_idx in POSE_SWAP_PAIRS:
            l_start, r_start = left_idx * 4, right_idx * 4
            buffer = flipped[l_start : l_start + 4].copy()
            flipped[l_start : l_start + 4] = flipped[r_start : r_start + 4]
            flipped[r_start : r_start + 4] = buffer

        # 3. Swap Hands (Block swap)
        left_hand_start = POSE_LEN + FACE_LEN
        right_hand_start = left_hand_start + HAND_LEN
        
        left_hand_data = flipped[left_hand_start : left_hand_start + HAND_LEN].copy()
        right_hand_data = flipped[right_hand_start : right_hand_start + HAND_LEN].copy()
        
        flipped[left_hand_start : left_hand_start + HAND_LEN] = right_hand_data
        flipped[right_hand_start : right_hand_start + HAND_LEN] = left_hand_data

        return flipped

    @classmethod
    def apply_rotation(cls, keypoints: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotates keypoints around the pose centroid."""
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        cx, cy = cls.get_centroid(keypoints)

        rotated = keypoints.copy()

        def _rotate_point(idx_x: int, idx_y: int) -> None:
            if rotated[idx_x] != 0:
                dx, dy = rotated[idx_x] - cx, rotated[idx_y] - cy
                rotated[idx_x] = dx * cos_a - dy * sin_a + cx
                rotated[idx_y] = dx * sin_a + dy * cos_a + cy

        # Rotate Pose
        for i in range(0, POSE_LEN, 4): 
            _rotate_point(i, i+1)
        # Rotate Face/Hands
        for i in range(POSE_LEN, len(rotated), 3): 
            _rotate_point(i, i+1)

        return rotated

    @classmethod
    def apply_translation(cls, keypoints: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Translates keypoints by (tx, ty)."""
        translated = keypoints.copy()
        
        # Helper to apply offset
        def _translate_point(idx_x: int, idx_y: int) -> None:
            if translated[idx_x] != 0:
                translated[idx_x] += tx
                translated[idx_y] += ty

        for i in range(0, POSE_LEN, 4):
            _translate_point(i, i+1)
        for i in range(POSE_LEN, len(translated), 3):
            _translate_point(i, i+1)
            
        return translated

    @classmethod
    def apply_scaling(cls, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """Scales keypoints relative to the pose centroid."""
        scaled = keypoints.copy()
        cx, cy = cls.get_centroid(keypoints)

        def _scale_point(idx_x: int, idx_y: int, idx_z: int) -> None:
            if scaled[idx_x] != 0:
                scaled[idx_x] = (scaled[idx_x] - cx) * scale + cx
                scaled[idx_y] = (scaled[idx_y] - cy) * scale + cy
                scaled[idx_z] *= scale

        for i in range(0, POSE_LEN, 4):
            _scale_point(i, i+1, i+2)
        for i in range(POSE_LEN, len(scaled), 3):
            _scale_point(i, i+1, i+2)

        return scaled


class DataCollector:
    """Manages webcam recording, dataset structure, and augmentation workflows."""

    def __init__(self, rewrite: bool = False, append: bool = False):
        self.rewrite = rewrite
        self.append = append
        self.cap = cv2.VideoCapture(0)

    def _setup_directories(self) -> None:
        """Initializes dataset directory structure."""
        if self.rewrite and os.path.exists(DATASET_DIR):
            shutil.rmtree(DATASET_DIR)
        
        os.makedirs(DATASET_DIR, exist_ok=True)
        for action in ACTIONS:
            os.makedirs(os.path.join(DATASET_DIR, action), exist_ok=True)

    def _generate_augmentations(self, action: str, sequence: int, is_flipped: bool) -> None:
        """Creates rotated, translated, and scaled versions of a sequence."""
        suffix = "_flipped" if is_flipped else ""
        base_dir = os.path.join(DATASET_DIR, action, f"{sequence}{suffix}")
        
        if not os.path.exists(base_dir):
            return

        # Augmentation paths
        aug_types = {
            "rot": os.path.join(DATASET_DIR, action, f"{sequence}{suffix}_aug_rot"),
            "trans": os.path.join(DATASET_DIR, action, f"{sequence}{suffix}_aug_trans"),
            "scale": os.path.join(DATASET_DIR, action, f"{sequence}{suffix}_aug_scale"),
        }

        # Skip if already exists
        if all(os.path.exists(p) and len(os.listdir(p)) >= SEQUENCE_LENGTH for p in aug_types.values()):
            return

        # Prepare directories and parameters
        for p in aug_types.values():
            os.makedirs(p, exist_ok=True)

        params = {
            'rot': np.random.uniform(-15, 15),
            'trans_x': np.random.uniform(-0.15, 0.15),
            'trans_y': np.random.uniform(-0.15, 0.15),
            'scale': np.random.uniform(0.85, 1.15)
        }

        # Process frame by frame
        for i in range(SEQUENCE_LENGTH):
            src_path = os.path.join(base_dir, f"{i}.npy")
            if not os.path.exists(src_path): 
                continue

            kp = np.load(src_path)
            
            np.save(os.path.join(aug_types["rot"], f"{i}.npy"), 
                    DataAugmenter.apply_rotation(kp, params['rot']))
            
            np.save(os.path.join(aug_types["trans"], f"{i}.npy"), 
                    DataAugmenter.apply_translation(kp, params['trans_x'], params['trans_y']))
            
            np.save(os.path.join(aug_types["scale"], f"{i}.npy"), 
                    DataAugmenter.apply_scaling(kp, params['scale']))

    def _extend_dataset(self) -> None:
        """Appends new sequences by copying and flipping existing data."""
        print("--- APPEND MODE ENABLED ---")
        for action in ACTIONS:
            action_path = os.path.join(DATASET_DIR, action)
            if not os.path.exists(action_path):
                continue

            existing_indices = [int(n) for n in os.listdir(action_path) if n.isdigit()]
            if not existing_indices:
                continue

            max_idx = max(existing_indices)
            start_new = max_idx + 1
            
            print(f"[APPEND] Extending {action}: IDs {start_new} to {start_new + NO_SEQUENCES - 1}")

            # Generate copies
            for tgt_idx in range(start_new, start_new + NO_SEQUENCES):
                src_idx = random.choice(existing_indices)
                src_dir = os.path.join(action_path, str(src_idx))
                
                tgt_dir = os.path.join(action_path, str(tgt_idx))
                tgt_dir_flip = os.path.join(action_path, f"{tgt_idx}_flipped")
                
                os.makedirs(tgt_dir, exist_ok=True)
                os.makedirs(tgt_dir_flip, exist_ok=True)

                for frame in range(SEQUENCE_LENGTH):
                    src_file = os.path.join(src_dir, f"{frame}.npy")
                    if os.path.exists(src_file):
                        kp = np.load(src_file)
                        np.save(os.path.join(tgt_dir, f"{frame}.npy"), kp)
                        np.save(os.path.join(tgt_dir_flip, f"{frame}.npy"), DataAugmenter.flip_keypoints(kp))
            
            # Generate augmentations for the new copies
            for seq in range(start_new, start_new + NO_SEQUENCES):
                self._generate_augmentations(action, seq, is_flipped=False)
                self._generate_augmentations(action, seq, is_flipped=True)
        
        print("--- Append completed. ---")

    def _show_countdown(self, action_name: str, sequence_num: int, duration: int = 3) -> bool:
        """Displays a visual countdown. Returns False if user cancels (presses Q)."""
        start_time = time.time()
        
        while int(time.time() - start_time) < duration:
            ret, frame = self.cap.read()
            if not ret: 
                return False
            
            seconds_left = duration - int(time.time() - start_time)
            
            # UI Overlay
            cv2.putText(frame, f'PREPARE FOR: {action_name}', (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Sequence: {sequence_num}', (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, str(seconds_left), (280, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5, cv2.LINE_AA)
            
            cv2.imshow('OpenCV Feed', frame)
            
            if cv2.waitKey(10) & 0xFF == ord("q"):
                return False
        return True

    def run(self) -> None:
        """Main execution loop."""
        self._setup_directories()

        if self.append:
            self._extend_dataset()
            return

        with mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        ) as holistic:
            
            for action in ACTIONS:
                for sequence in range(NO_SEQUENCES):
                    dir_path = os.path.join(DATASET_DIR, action, str(sequence))
                    dir_path_flip = os.path.join(DATASET_DIR, action, f"{sequence}_flipped")

                    # Skip if exists and not rewriting
                    if not self.rewrite and os.path.exists(dir_path):
                        self._generate_augmentations(action, sequence, False)
                        self._generate_augmentations(action, sequence, True)
                        print(f"[SKIP] {action} #{sequence} exists.")
                        continue

                    # Create directories
                    os.makedirs(dir_path, exist_ok=True)
                    os.makedirs(dir_path_flip, exist_ok=True)

                    # Countdown
                    wait_time = 4 if sequence == 0 else 1
                    if not self._show_countdown(action, sequence, wait_time):
                        print("[INFO] User cancelled.")
                        return

                    # Recording Loop
                    print(f"[REC] {action} #{sequence}")
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = self.cap.read()
                        if not ret: break

                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)

                        # Status Overlay
                        cv2.putText(image, f'REC: {action} #{sequence}', (15, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.putText(image, f"Frame: {frame_num}/{SEQUENCE_LENGTH}", (15, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                        
                        cv2.imshow('OpenCV Feed', image)

                        # Save Keypoints
                        kp = extract_keypoints(results)
                        np.save(os.path.join(dir_path, str(frame_num)), kp)
                        np.save(os.path.join(dir_path_flip, str(frame_num)), DataAugmenter.flip_keypoints(kp))

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("[INFO] User cancelled.")
                            return

                    # Post-recording augmentation
                    self._generate_augmentations(action, sequence, False)
                    self._generate_augmentations(action, sequence, True)

        self.cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sign Language Data Collection")
    parser.add_argument("-R", "--rewrite", action="store_true", help="Delete and rebuild dataset.")
    parser.add_argument("-a", "--append", action="store_true", help="Append copies of existing data.")
    args = parser.parse_args()

    collector = DataCollector(rewrite=args.rewrite, append=args.append)
    try:
        collector.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        if collector.cap.isOpened():
            collector.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()