import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Final

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from config.dynamic import DATA_PATH, NO_SEQUENCES, SEQUENCE_LENGTH, TEST_SIZE
from config.gestures import DYNAMIC_ACTIONS

# --- CONSTANTS ---
ACTIONS: Final[np.ndarray] = DYNAMIC_ACTIONS
AUGMENTATION_SUFFIXES: Final[List[str]] = ['_aug_rot', '_aug_trans', '_aug_scale']


class DataPreprocessor:
    """
    Handles data loading and splitting with strict anti-leakage policies.
    Loads Original + Standard Augmentations (Rotation, Translation, Scale).
    Explicitly EXCLUDES flipped data to maintain anatomical consistency if required.
    """

    @staticmethod
    def _create_label_map() -> Dict[str, int]:
        """Maps action strings to integer labels."""
        return {label: num for num, label in enumerate(ACTIONS)}

    @staticmethod
    def _load_single_sequence(path: str, sequence_length: int) -> Optional[np.ndarray]:
        """
        Loads a single video sequence from .npy files.
        Returns None if the sequence is incomplete or missing.
        """
        if not os.path.exists(os.path.join(path, "0.npy")):
            return None

        try:
            window = []
            for frame_num in range(sequence_length):
                file_path = os.path.join(path, f"{frame_num}.npy")
                frame_data = np.load(file_path)
                window.append(frame_data)
            return np.array(window)
        except (OSError, ValueError) as e:
            # Catching specific errors is better than generic Exception
            print(f"Error loading sequence at {path}: {e}")
            return None

    @staticmethod
    def _get_sequence_paths(action: str, sequence_id: int, label: int) -> List[Tuple[str, int]]:
        """
        Generates file paths for a sequence ID and its standard augmentations.
        Strictly excludes 'flipped' variants.
        """
        tasks = []
        base_path = os.path.join(DATA_PATH, action, str(sequence_id))

        # 1. Original Sequence
        tasks.append((base_path, label))

        # 2. Standard Augmentations
        for suffix in AUGMENTATION_SUFFIXES:
            aug_path = base_path + suffix
            # Note: We rely on the loader to return None if path doesn't exist,
            # avoiding expensive OS calls here.
            tasks.append((aug_path, label))

        return tasks

    @classmethod
    def load_dataset_split(cls) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Orchestrates the train/test split and data loading.
        
        Splitting Strategy: Sequence ID-based.
        All variants (original, rot, scale) of a specific Sequence ID stay together
        in either Train or Test to prevent data leakage.
        """
        label_map = cls._create_label_map()
        
        # Split sequence IDs (e.g., 0-199)
        all_sequence_ids = np.arange(NO_SEQUENCES)
        train_ids, test_ids = train_test_split(
            all_sequence_ids, 
            test_size=TEST_SIZE, 
            random_state=42
        )

        print(f"[INFO] Split Strategy: Sequence ID-based")
        print(f"[INFO] Train Sequences: {len(train_ids)} | Test Sequences: {len(test_ids)}")

        def _build_tasks(ids_list: np.ndarray) -> List[Tuple[str, int]]:
            tasks = []
            for action in ACTIONS:
                label = label_map[action]
                for seq_id in ids_list:
                    tasks.extend(cls._get_sequence_paths(action, seq_id, label))
            return tasks

        train_tasks = _build_tasks(train_ids)
        test_tasks = _build_tasks(test_ids)

        def _execute_loading(tasks_list: List[Tuple[str, int]], desc: str) -> Tuple[np.ndarray, np.ndarray]:
            print(f"[INFO] Loading {desc} dataset ({len(tasks_list)} paths)...")
            X_data, y_data = [], []

            with ThreadPoolExecutor() as executor:
                # Partial application to fix sequence length argument
                loader = functools.partial(cls._load_single_sequence, sequence_length=SEQUENCE_LENGTH)
                
                paths = [t[0] for t in tasks_list]
                labels = [t[1] for t in tasks_list]

                results = executor.map(loader, paths)

                for res, lab in zip(results, labels):
                    if res is not None and len(res) == SEQUENCE_LENGTH:
                        X_data.append(res)
                        y_data.append(lab)

            return np.array(X_data), np.array(y_data)

        # Execute Parallel Loading
        X_train, y_train_idx = _execute_loading(train_tasks, "TRAIN")
        X_test, y_test_idx = _execute_loading(test_tasks, "TEST")

        # One-hot encoding
        num_classes = len(ACTIONS)
        y_train = to_categorical(y_train_idx, num_classes=num_classes).astype(int)
        y_test = to_categorical(y_test_idx, num_classes=num_classes).astype(int)

        return X_train, X_test, y_train, y_test


def preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Public interface for dataset loading."""
    return DataPreprocessor.load_dataset_split()