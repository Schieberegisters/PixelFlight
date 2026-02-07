import sys
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

from DroneControl.gestureRecognition import calculate_distance, landmark_normalization

class TestGestureUtils:

    def test_calculate_distance_logic(self):
        """Verify Euclidean distance for a standard 3-4-5 triangle."""
        # Distance between (0,0) and (3,4) should be 5.0
        assert calculate_distance(0, 0, 3, 4) == 5.0
        # Distance between (1.1, 2.2) and (1.1, 2.2) should be 0.0
        assert calculate_distance(1.1, 2.2, 1.1, 2.2) == 0.0

    def test_landmark_normalization_feature_count(self):
        """Verify that the function returns exactly 15 distance-based features."""
        # Create 21 MagicMocks to simulate MediaPipe landmarks
        mock_hand = [MagicMock(x=0.1, y=0.1) for _ in range(21)]
        
        features = landmark_normalization(mock_hand)
        
        assert isinstance(features, list)
        # 5 (wrist-to-tip) + 10 (pairwise tip-to-tip) = 15 features
        assert len(features) == 15

    def test_landmark_normalization_math_accuracy(self):
        """Verify specific distance calculations using MagicMocks."""
        # Initialize 21 mocks at origin (0,0)
        mock_hand = [MagicMock(x=0.0, y=0.0) for _ in range(21)]
        
        # Wrist (0) is at (0,0)
        # Set thumb tip (index 4) at (1,0)
        mock_hand[4].x = 1.0
        mock_hand[4].y = 0.0
        
        # Set index tip (index 8) at (0,1)
        mock_hand[8].x = 0.0
        mock_hand[8].y = 1.0
        
        features = landmark_normalization(mock_hand)
        
        # 1. Wrist (0) to Thumb Tip (4) distance should be 1.0
        assert features[0] == 1.0 
        
        # 2. Wrist (0) to Index Tip (8) distance should be 1.0
        assert features[1] == 1.0
        
        # 3. Pairwise: Thumb Tip (4) to Index Tip (8)
        # Using sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        # This is the 6th feature (index 5) in the returned list
        expected_pair_dist = math.sqrt(2)
        assert math.isclose(features[5], expected_pair_dist, rel_tol=1e-7)

    def test_landmark_normalization_missing_landmarks(self):
        """Verify it raises IndexError if the hand landmark list is incomplete."""
        # Mediapipe hands must have 21 landmarks; test behavior with fewer
        incomplete_hand = [MagicMock(x=0.0, y=0.0) for _ in range(5)]
        with pytest.raises(IndexError):
            landmark_normalization(incomplete_hand)

    def test_landmark_normalization_property_access(self):
        """Verify that the function correctly reads the .x and .y attributes."""
        mock_point = MagicMock()
        
        # Use PropertyMock to track access events
        # This tells the mock: "When someone asks for .x, return 0.5, but record the event."
        p = PropertyMock(return_value=0.5)
        type(mock_point).x = p
        type(mock_point).y = p
        
        mock_hand = [mock_point for _ in range(21)]
        
        landmark_normalization(mock_hand)
        
        # Verify the code actually triggered the property read
        assert p.call_count > 0


def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()