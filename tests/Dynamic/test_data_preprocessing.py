import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import the module object
from Dynamic import data_preprocessing

# Import specific components
from Dynamic.data_preprocessing import (
    DataPreprocessor,
    ACTIONS,
    SEQUENCE_LENGTH,
    AUGMENTATION_SUFFIXES
)

class TestDataPreprocessing:

    # --- UNIT TESTS ---

    def test_create_label_map(self):
        """Verify label mapping matches the ACTIONS constant order."""
        label_map = DataPreprocessor._create_label_map()
        
        assert isinstance(label_map, dict)
        assert len(label_map) == len(ACTIONS)
        
        # Check first and last action mapping
        assert label_map[ACTIONS[0]] == 0
        assert label_map[ACTIONS[-1]] == len(ACTIONS) - 1

    def test_load_single_sequence_success(self):
        """Verify loading a complete sequence of .npy files."""
        fake_path = "fake/path/to/seq"
        dummy_frame = np.zeros(1662)
        
        with patch("os.path.exists", return_value=True), \
             patch("numpy.load", return_value=dummy_frame) as mock_load:
            
            result = DataPreprocessor._load_single_sequence(fake_path, SEQUENCE_LENGTH)
            
            assert result is not None
            assert result.shape == (SEQUENCE_LENGTH, 1662)
            assert mock_load.call_count == SEQUENCE_LENGTH

    def test_load_single_sequence_missing_dir(self):
        """Verify returns None if the directory or 0.npy doesn't exist."""
        with patch("os.path.exists", return_value=False):
            result = DataPreprocessor._load_single_sequence("bad/path", 10)
            assert result is None

    def test_load_single_sequence_corrupt_file(self):
        """Verify graceful handling of IO errors during loading."""
        with patch("os.path.exists", return_value=True), \
             patch("numpy.load", side_effect=ValueError("Corrupt")):
            
            result = DataPreprocessor._load_single_sequence("path", 10)
            assert result is None

    def test_get_sequence_paths_logic(self):
        """Verify path generation logic (Original + Augments, NO Flips)."""
        action = "TEST_ACTION"
        seq_id = 1
        label = 0
        
        tasks = DataPreprocessor._get_sequence_paths(action, seq_id, label)
        
        # Expected: 1 original + 3 augmentations = 4 tasks
        assert len(tasks) == 1 + len(AUGMENTATION_SUFFIXES)
        
        paths = [t[0] for t in tasks]
        
        # Check specific suffixes
        base_str = str(Path(data_preprocessing.DATASET_DIR) / action / str(seq_id))
        
        assert any(p == base_str for p in paths) # Original
        assert any("aug_rot" in p for p in paths)
        assert any("aug_trans" in p for p in paths)
        
        # CRITICAL: Ensure NO flipped paths are included
        assert not any("flipped" in p for p in paths)

    # --- INTEGRATION TEST (WORKFLOW) ---

    def test_load_dataset_split_integration(self):
        """
        Tests the full orchestration:
        ID Split -> Task Building -> Parallel Loading -> One-Hot Encoding.
        """
        # 1. Mock IDs
        mock_train_ids = np.array([0, 1])
        mock_test_ids = np.array([2])
        
        # 2. Mock loader output
        dummy_seq = np.zeros((SEQUENCE_LENGTH, 1662))
        
        # FIX: Patch 'train_test_split' where it is USED, not where it is defined.
        with patch("Dynamic.data_preprocessing.train_test_split", return_value=(mock_train_ids, mock_test_ids)), \
             patch.object(DataPreprocessor, "_load_single_sequence", return_value=dummy_seq), \
             patch("Dynamic.data_preprocessing.ACTIONS", ["ACT_A", "ACT_B"]): 
            
            # Expected Count Calculation:
            # Actions: 2 ("ACT_A", "ACT_B")
            # Variants per ID: 4 (1 Original + 3 Augments)
            # Train IDs: 2 ([0, 1]) -> 2 * 2 * 4 = 16 samples
            # Test IDs:  1 ([2])    -> 2 * 1 * 4 = 8 samples
            
            X_train, X_test, y_train, y_test = DataPreprocessor.load_dataset_split()
            
            # Check Shapes
            assert X_train.shape == (16, SEQUENCE_LENGTH, 1662)
            assert X_test.shape == (8, SEQUENCE_LENGTH, 1662)
            
            # Check Labels (One-hot with 2 classes)
            assert y_train.shape == (16, 2)
            assert y_test.shape == (8, 2)
            
            # Verify data types
            assert X_train.dtype == np.float64
            assert y_train.dtype == int

def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __name__ if __name__ != "__main__" else __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()