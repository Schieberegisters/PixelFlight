import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Iterable, Tuple

import numpy as np

# --- CONFIGURATION & PATHS ---
TRAINING_DATA_DIR: Final[Path] = Path(__file__).resolve().parent / "TrainingData"
SOURCE_DIR: Final[Path] = TRAINING_DATA_DIR / "DynamicRecognition"
DEST_DIR: Final[Path] = TRAINING_DATA_DIR / "DynamicRecognition_relational"

# --- DATA LAYOUT CONSTANTS ---
# Segments: Pose (33*4), Face (468*3), Left Hand (21*3), Right Hand (21*3)
POSE_LEN: Final[int] = 132    # 33 * 4
FACE_LEN: Final[int] = 1404   # 468 * 3
HAND_LEN: Final[int] = 63     # 21 * 3
TOTAL_LEN: Final[int] = POSE_LEN + FACE_LEN + (HAND_LEN * 2) # 1662

NOSE_SLICE: Final[slice] = slice(0, 3)


def convert_relative_vectorized(keypoints: np.ndarray) -> np.ndarray:
    """Converts absolute keypoint coordinates to nose-relative coordinates."""
    relative = keypoints.copy()
    nose = keypoints[NOSE_SLICE]

    # If nose is not detected (all zeros), return original as fallback
    if not np.any(nose):
        return relative

    # 1. Process Pose (x, y, z, visibility)
    # Reshape to (33, 4) to isolate x,y,z columns
    pose_view = relative[:POSE_LEN].reshape(-1, 4)
    pose_xyz = pose_view[:, :3]
    
    # Subtract nose position only where landmarks exist (non-zero)
    mask_pose = np.any(pose_xyz != 0, axis=1)
    pose_xyz[mask_pose] -= nose

    # 2. Process Face & Hands (x, y, z)
    # Reshape remaining elements to (N, 3)
    rest_view = relative[POSE_LEN:].reshape(-1, 3)
    
    mask_rest = np.any(rest_view != 0, axis=1)
    rest_view[mask_rest] -= nose

    # Reset nose itself to 0,0,0 as it is the origin
    relative[NOSE_SLICE] = 0.0
    return relative


def process_file(task: Tuple[Path, Path]) -> int:
    """Loads, converts, and saves a single .npy file."""
    source, dest = task
    try:
        data = np.load(source)
        if data.shape != (TOTAL_LEN,):
            return 0
        
        converted = convert_relative_vectorized(data)
        np.save(dest, converted)
        return 1
    except Exception as e:
        print(f"Error processing {source.name}: {e}")
        return 0


def generate_tasks(source_root: Path, dest_root: Path, skip_flipped: bool) -> Iterable[Tuple[Path, Path]]:
    """Yields source/destination pairs, mirroring the directory structure."""
    for path in source_root.rglob("*.npy"):
        if skip_flipped and "flipped" in path.name:
            continue
            
        rel_path = path.relative_to(source_root)
        target_path = dest_root / rel_path
        
        # Ensure parent directory exists before yielding
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        yield path, target_path


def parse_arguments() -> argparse.Namespace:
    """Configures CLI arguments."""
    parser = argparse.ArgumentParser(description="Normalize dataset to nose-relative coordinates.")
    parser.add_argument(
        "-n", "--no-flipped", 
        action="store_true", 
        help="Exclude augmented 'flipped' files from processing."
    )
    return parser.parse_args()


def main() -> None:
    """Main execution controller."""
    args = parse_arguments()

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print("Scanning for files...")
    tasks = list(generate_tasks(SOURCE_DIR, DEST_DIR, args.no_flipped))
    
    if not tasks:
        print("No .npy files found.")
        return

    print(f"Processing {len(tasks)} files with {os.cpu_count() or 4} threads...")
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, tasks))

    success_count = sum(results)
    print(f"Done. Processed: {success_count}/{len(tasks)}.")
    print(f"Output: {DEST_DIR}")


if __name__ == "__main__":
    main()