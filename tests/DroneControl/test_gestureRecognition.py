import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from DroneControl.gestureRecognition import calculate_distance, landmark_normalization

class TestGestureUtils:
    """Tests utility functions for gesture distance and normalization."""

    def test_calculate_distance_logic(self) -> None:
        """Verifies Euclidean distance for a standard 3-4-5 triangle and zero distance."""
        # Standard Euclidean distance between (0,0) and (3,4)
        assert calculate_distance(0, 0, 3, 4) == 5.0
        # Zero distance between identical points
        assert calculate_distance(1.1, 2.2, 1.1, 2.2) == 0.0

    def test_landmark_normalization_feature_count(self) -> None:
        """Ensures the function returns exactly 15 distance-based features."""
        # MediaPipe hands consist of 21 landmarks
        mock_hand = [MagicMock(x=0.1, y=0.1) for _ in range(21)]
        
        features = landmark_normalization(mock_hand)
        
        assert isinstance(features, list)
        # Expected: 5 (wrist-to-tip) + 10 (pairwise tip-to-tip) = 15 features
        assert len(features) == 15

    def test_landmark_normalization_math_accuracy(self) -> None:
        """Validates specific distance calculations using defined mock coordinates."""
        mock_hand = [MagicMock(x=0.0, y=0.0) for _ in range(21)]
        
        # Wrist (0) at origin, Thumb tip (4) at (1,0), Index tip (8) at (0,1)
        mock_hand[4].x = 1.0
        mock_hand[4].y = 0.0
        mock_hand[8].x = 0.0
        mock_hand[8].y = 1.0
        
        features = landmark_normalization(mock_hand)
        
        # Wrist (0) to Thumb Tip (4) distance
        assert features[0] == 1.0 
        
        # Wrist (0) to Index Tip (8) distance
        assert features[1] == 1.0
        
        # Pairwise: Thumb Tip (4) to Index Tip (8) distance (sqrt(2))
        expected_pair_dist = math.sqrt(2)
        assert math.isclose(features[5], expected_pair_dist, rel_tol=1e-7)

    def test_landmark_normalization_missing_landmarks(self) -> None:
        """Verifies that an incomplete landmark list raises an IndexError."""
        incomplete_hand = [MagicMock(x=0.0, y=0.0) for _ in range(5)]
        with pytest.raises(IndexError):
            landmark_normalization(incomplete_hand)

    def test_landmark_normalization_property_access(self) -> None:
        """Ensures the function correctly accesses landmark attributes."""
        mock_point = MagicMock()
        
        prop_mock = PropertyMock(return_value=0.5)
        type(mock_point).x = prop_mock
        type(mock_point).y = prop_mock
        
        mock_hand = [mock_point for _ in range(21)]
        landmark_normalization(mock_hand)
        
        assert prop_mock.call_count > 0

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