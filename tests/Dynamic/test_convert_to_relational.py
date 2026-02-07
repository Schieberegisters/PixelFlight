import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    # Standard setup to allow direct execution
    sys.path.append(str(Path(__file__).parents[2]))

# Import the module OBJECT (required for stable patching)
from Dynamic import convert_to_relational

# Import specific constants and functions for the test logic
from Dynamic.convert_to_relational import (
    convert_relative_vectorized,
    process_file,
    generate_tasks,
    TOTAL_LEN,
    POSE_LEN
)

class TestDatasetNormalization:

    def test_convert_relative_math_logic(self):
        """Verify the nose-relative subtraction logic works correctly."""
        data = np.zeros(TOTAL_LEN)
        
        # 1. Setup Nose (Reference Point) at (10, 20, 30)
        data[0] = 10.0; data[1] = 20.0; data[2] = 30.0
        
        # 2. Setup a target point in POSE section (e.g., Left Eye)
        # Point at (15, 25, 35) -> Relative should be (5, 5, 5)
        data[4] = 15.0; data[5] = 25.0; data[6] = 35.0
        data[7] = 0.9 # Visibility channel (must remain untouched)
        
        # 3. Setup a target point in FACE/HAND section
        idx = POSE_LEN
        data[idx] = 12.0; data[idx+1] = 22.0; data[idx+2] = 32.0

        result = convert_relative_vectorized(data)

        # Assertions
        assert result[0] == 0.0 # Nose itself becomes origin
        assert result[4] == 5.0 # Pose X: 15 - 10
        assert result[7] == 0.9 # Visibility is not modified
        assert result[idx] == 2.0 # Face/Hand X: 12 - 10

    def test_convert_relative_undetected_nose(self):
        """If nose is (0,0,0), return data unchanged as a fallback."""
        data = np.zeros(TOTAL_LEN)
        data[10] = 55.0
        result = convert_relative_vectorized(data)
        np.testing.assert_array_equal(data, result)

    def test_convert_relative_zero_padding(self):
        """Undetected points (0,0,0) should remain (0,0,0), not become negative nose values."""
        data = np.zeros(TOTAL_LEN)
        data[0] = 10.0 # Nose exists at X=10
        
        # Point at index 20 is (0,0,0) -> i.e., MediaPipe lost track
        result = convert_relative_vectorized(data)
        
        # Should NOT become -10.0
        assert result[20] == 0.0

    def test_process_file_success(self):
        """Test successful loading, converting, and saving cycle."""
        mock_source = MagicMock(spec=Path)
        mock_dest = MagicMock(spec=Path)
        dummy_data = np.zeros(TOTAL_LEN)

        # Use patch.object on the imported module to avoid string-path resolution errors
        with patch("numpy.load", return_value=dummy_data) as mock_load, \
             patch("numpy.save") as mock_save, \
             patch.object(convert_to_relational, "convert_relative_vectorized") as mock_convert:
            
            mock_convert.return_value = dummy_data
            
            result = process_file((mock_source, mock_dest))
            
            assert result == 1
            mock_load.assert_called_once_with(mock_source)
            mock_convert.assert_called_once()
            mock_save.assert_called_once_with(mock_dest, dummy_data)

    def test_process_file_wrong_shape(self):
        """Test that files with incorrect numpy shapes are ignored."""
        mock_source = MagicMock(spec=Path)
        mock_dest = MagicMock(spec=Path)
        bad_data = np.zeros(100) # Incorrect length
        
        with patch("numpy.load", return_value=bad_data), \
             patch("numpy.save") as mock_save:
            
            result = process_file((mock_source, mock_dest))
            assert result == 0
            mock_save.assert_not_called()

    def test_process_file_exception_handling(self):
        """Test that IO errors (e.g. corrupt .npy) are handled without crashing."""
        mock_source = MagicMock(spec=Path)
        mock_source.name = "corrupt_data.npy"
        
        with patch("numpy.load", side_effect=ValueError("Corrupt")):
            result = process_file((mock_source, MagicMock()))
            assert result == 0

    def test_generate_tasks_traversal(self):
        """Test recursive directory scanning and target path creation."""
        mock_root = MagicMock(spec=Path)
        mock_dest = MagicMock(spec=Path)
        
        file1 = MagicMock(spec=Path)
        file1.name = "sample.npy"
        file1.relative_to.return_value = Path("class_01/sample.npy")
        mock_root.rglob.return_value = [file1]
        
        tasks = list(generate_tasks(mock_root, mock_dest, skip_flipped=False))
        
        assert len(tasks) == 1
        source, target = tasks[0]
        assert target == mock_dest / Path("class_01/sample.npy")
        # Verify the parent directory is created for the target
        assert target.parent.mkdir.called

    def test_generate_tasks_skip_flipped(self):
        """Verify the filter for augmented 'flipped' files."""
        mock_root = MagicMock(spec=Path)
        f1 = MagicMock(spec=Path); f1.name = "original.npy"
        f2 = MagicMock(spec=Path); f2.name = "original_flipped.npy"
        mock_root.rglob.return_value = [f1, f2]
        
        tasks = list(generate_tasks(mock_root, MagicMock(), skip_flipped=True))
        
        assert len(tasks) == 1
        assert tasks[0][0].name == "original.npy"

def run_tests_directly() -> None:
    """Standard entry point for local execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __name__ if __name__ != "__main__" else __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()