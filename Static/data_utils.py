from __future__ import annotations

import copy
import csv
import os
import math
from collections.abc import Sequence
from typing import Any, Final, List, Dict, Optional, Protocol, Tuple, Union

import cv2
import numpy as np

# --- TYPES ---
class HasXY(Protocol):
    """Protocol for objects with x and y coordinates (e.g., MediaPipe landmarks)."""
    x: float
    y: float

# --- GEOMETRY UTILITIES ---

def flip_frame(frame: np.ndarray) -> np.ndarray:
    """Horizontally flips a frame for natural webcam interaction."""
    return cv2.flip(frame, 1)


def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes Euclidean distance between two 2D points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def normalize_landmarks(hand_points: Sequence[HasXY]) -> List[float]:
    """
    Computes a 15-dimensional feature vector based on Euclidean distances.
    Features:
    - 5 distances: Wrist to each fingertip.
    - 10 distances: Pairwise distances between adjacent fingertips.
    """
    wrist = hand_points[0]
    fingertip_indices: Final[Tuple[int, ...]] = (4, 8, 12, 16, 20)

    distances: List[float] = []

    # 1. Wrist to Fingertips
    for idx in fingertip_indices:
        tip = hand_points[idx]
        distances.append(calculate_distance(wrist.x, wrist.y, tip.x, tip.y))

    # 2. Pairwise Fingertips
    for i in range(len(fingertip_indices)):
        for j in range(i + 1, len(fingertip_indices)):
            tip1 = hand_points[fingertip_indices[i]]
            tip2 = hand_points[fingertip_indices[j]]
            distances.append(calculate_distance(tip1.x, tip1.y, tip2.x, tip2.y))

    return distances


# --- DATA AUGMENTATION ---

def scale_sequence_data(
    sequence_data: List[Dict[str, Any]],
    factors: Sequence[float],
    next_seq_id_start: int,
) -> Tuple[List[List[Dict[str, Any]]], int]:
    """Generates scaled variants of a sequence for data augmentation."""
    scaled_sequences = []
    current_id = next_seq_id_start
    
    # Columns strictly to exclude from scaling
    immutable_cols = {'label', 'seq_id'}

    for factor in factors:
        new_sequence = []
        
        for frame in sequence_data:
            # Deep copy to ensure isolation
            new_frame = copy.deepcopy(frame)
            new_frame['seq_id'] = str(current_id)

            for key, val in new_frame.items():
                if key not in immutable_cols:
                    try:
                        # Attempt to scale numeric feature columns
                        original_val = float(val)
                        new_frame[key] = f"{original_val * factor:.6f}"
                    except (ValueError, TypeError):
                        continue

            new_sequence.append(new_frame)

        scaled_sequences.append(new_sequence)
        current_id += 1

    return scaled_sequences, current_id


# --- CSV I/O ---

def save_data_to_csv(
    data_rows: Union[Sequence[Sequence[Any]], Sequence[Dict[str, Any]]],
    csv_path: str,
    mode: str = "a",
    write_header: bool = False,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    """Writes rows (dict or list) to a CSV file."""
    if not data_rows:
        print("Warning: No data to save.")
        return

    # check if file is empty to determine if header is needed
    file_is_empty = False
    if mode == "a" and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as f:
                file_is_empty = (f.read(1) == "")
        except OSError:
            pass
    
    should_write_header = write_header or file_is_empty

    with open(csv_path, mode, newline="") as file:
        first_row = data_rows[0]
        
        # Handle Dictionary Rows
        if isinstance(first_row, dict):
            # Infer fieldnames from keys if not provided
            effective_fieldnames = list(fieldnames) if fieldnames else list(first_row.keys())
            writer = csv.DictWriter(file, fieldnames=effective_fieldnames)
            
            if should_write_header:
                writer.writeheader()
            writer.writerows(data_rows) # type: ignore
            return

        # Handle List Rows
        writer = csv.writer(file)
        if should_write_header and fieldnames:
            writer.writerow(list(fieldnames))
        writer.writerows(data_rows) # type: ignore


def load_sequences_from_csv(csv_path: str) -> Tuple[Dict[int, List[Dict[str, Any]]], List[str]]:
    """Loads CSV data and groups frames by `seq_id`."""
    sequences: Dict[int, List[Dict[str, Any]]] = {}
    fieldnames: List[str] = []

    with open(csv_path, newline='', encoding='utf-8') as file:
        # Heuristic to detect header presence
        first_line = file.readline().strip()
        file.seek(0)
        
        has_header = ('_' in first_line) or ('rel_' in first_line)

        if has_header:
            reader = csv.DictReader(file)
            fieldnames = reader.fieldnames if reader.fieldnames else []
            data_iterable = reader
        else:
            # Fallback for legacy CSVs without headers
            reader = csv.reader(file)
            fieldnames = _generate_legacy_fieldnames()
            data_iterable = reader

        for row in data_iterable:
            # Normalization to Dict
            if has_header:
                row_dict = row # type: ignore
            else:
                if len(row) != len(fieldnames):
                    continue
                row_dict = dict(zip(fieldnames, row)) # type: ignore

            try:
                seq_id = int(row_dict["seq_id"])
                if seq_id not in sequences:
                    sequences[seq_id] = []
                sequences[seq_id].append(row_dict)
            except (KeyError, ValueError):
                continue

    return sequences, fieldnames


def _generate_legacy_fieldnames() -> List[str]:
    """Generates default fieldnames for headerless CSVs (21 landmarks * 6 coords)."""
    names = []
    for i in range(21):
        names.extend([
            f'rel_x_{i}', f'rel_y_{i}', f'rel_z_{i}',
            f'delta_x_{i}', f'delta_y_{i}', f'delta_z_{i}'
        ])
    names.extend(['label', 'seq_id'])
    return names


def expand_training_data(
    csv_path: str,
    output_path: Optional[str] = None,
    factors: Optional[Sequence[float]] = None,
) -> Optional[str]:
    """
    Expands a training dataset by generating scaled augmented copies.
    """
    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = f"{base}_expanded.csv"

    if factors is None:
        factors = [0.8, 0.9, 1.1, 1.2, 1.3]

    print(f"Expanding dataset: {csv_path}")
    print(f"Augmentation factors: {factors}")

    original_seqs, fieldnames = load_sequences_from_csv(csv_path)
    if not original_seqs:
        print("Error: No valid sequences found.")
        return None

    # Determine next ID to avoid collisions
    next_id = max(original_seqs.keys()) + 1
    
    # Collect all rows
    all_rows = []
    
    # 1. Add Originals
    for frames in original_seqs.values():
        all_rows.extend(frames)

    # 2. Generate Augmentations
    count_aug = 0
    for frames in original_seqs.values():
        augmented_seqs, next_id = scale_sequence_data(frames, factors, next_id)
        for seq in augmented_seqs:
            all_rows.extend(seq)
            count_aug += 1

    # Save
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"--- Expansion Complete ---")
    print(f"Originals: {len(original_seqs)} | Augmented: {count_aug}")
    print(f"Total Sequences: {len(original_seqs) + count_aug}")
    
    return output_path
