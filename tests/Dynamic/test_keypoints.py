import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from Dynamic.keypoints import extract_keypoints, TOTAL_LEN, POSE_LEN, FACE_LEN, HAND_LEN

class TestKeypoints:
    @pytest.fixture
    def mock_results(self) -> MagicMock:
        """Creates a mock results object simulating MediaPipe landmark structures."""
        results = MagicMock()
        
        def create_landmarks(count: int, has_visibility: bool = False) -> MagicMock:
            landmarks = []
            for i in range(count):
                lm = MagicMock()
                lm.x = 0.1 * i
                lm.y = 0.2 * i
                lm.z = 0.3 * i
                if has_visibility:
                    lm.visibility = 0.9
                landmarks.append(lm)
            
            mock_obj = MagicMock()
            mock_obj.landmark = landmarks
            return mock_obj

        results.pose_landmarks = create_landmarks(33, has_visibility=True)
        results.face_landmarks = create_landmarks(468)
        results.left_hand_landmarks = create_landmarks(21)
        results.right_hand_landmarks = create_landmarks(21)
        
        return results

    def test_extract_keypoints_full_detection(self, mock_results: MagicMock) -> None:
        """Verifies extraction when all landmarks are detected."""
        kp = extract_keypoints(mock_results)
        
        assert isinstance(kp, np.ndarray)
        assert kp.shape == (TOTAL_LEN,)
        # Validate first landmark (Pose) values: x=0.0, y=0.0, z=0.0, v=0.9
        assert kp[0] == 0.0
        assert kp[3] == 0.9

    def test_extract_keypoints_partial_detection(self, mock_results: MagicMock) -> None:
        """Verifies that missing landmarks are zero-padded."""
        mock_results.face_landmarks = None
        mock_results.left_hand_landmarks = None
        
        kp = extract_keypoints(mock_results)
        
        assert kp.shape == (TOTAL_LEN,)
        
        # Pose segment must contain data
        assert np.any(kp[:POSE_LEN] != 0)
        
        # Face segment must be entirely zero
        face_segment = kp[POSE_LEN : POSE_LEN + FACE_LEN]
        assert np.all(face_segment == 0)
        
        # Right hand segment (tail) must contain data
        assert np.any(kp[-HAND_LEN:] != 0)

    def test_extract_keypoints_all_none(self) -> None:
        """Verifies edge case where no landmarks are detected."""
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_results.face_landmarks = None
        mock_results.left_hand_landmarks = None
        mock_results.right_hand_landmarks = None
        
        kp = extract_keypoints(mock_results)
        
        assert kp.shape == (TOTAL_LEN,)
        assert np.all(kp == 0)

    def test_extract_keypoints_shape_mismatch_error(self, mock_results: MagicMock) -> None:
        """Verifies ValueError is raised on unexpected landmark counts."""
        bad_face = MagicMock()
        bad_face.landmark = [MagicMock()] * 10
        mock_results.face_landmarks = bad_face
        
        with pytest.raises(ValueError, match="Vector shape mismatch"):
            extract_keypoints(mock_results)

def run_tests_directly() -> None:
    """Execution entry point for direct script testing."""
    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()