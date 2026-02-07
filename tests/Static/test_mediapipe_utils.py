import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import the module object
from Static import mediapipe_utils
from Static.mediapipe_utils import (
    get_hand_landmarker,
    process_frame_with_mediapipe,
    draw_hand_landmarks
)

class TestMediaPipeStaticUtils:

    @pytest.fixture
    def mock_landmarker_class(self):
        """Mocks the HandLandmarker class and its factory method."""
        with patch("Static.mediapipe_utils.HandLandmarker") as mock:
            yield mock

    @pytest.fixture
    def mock_mp_image(self):
        """Mocks the mp.Image class."""
        with patch("Static.mediapipe_utils.mp.Image") as mock:
            yield mock

    # --- INITIALIZATION TESTS ---

    def test_get_hand_landmarker_defaults(self, mock_landmarker_class):
        """Verify initialization with default parameters."""
        # Mock Path.exists to return True for the default model path
        with patch("pathlib.Path.exists", return_value=True):
            landmarker = get_hand_landmarker()
            
            # Check if create_from_options was called
            mock_landmarker_class.create_from_options.assert_called_once()
            
            # Inspect the options passed to create_from_options
            options = mock_landmarker_class.create_from_options.call_args[0][0]
            assert options.num_hands == 1
            assert options.min_hand_detection_confidence == 0.3
            # Running mode VIDEO is default
            # Note: We can't easily check the enum value equality without importing RunningMode,
            # but we can check if it passed *something*.

    def test_get_hand_landmarker_custom_path_missing(self):
        """Verify FileNotFoundError is raised if model path is invalid."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                get_hand_landmarker(model_path="non_existent.task")

    def test_get_hand_landmarker_invalid_mode(self):
        """Verify ValueError for invalid running modes."""
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(ValueError, match="Invalid running_mode"):
                get_hand_landmarker(running_mode="INVALID_MODE")

    # --- PROCESSING TESTS ---

    def test_process_frame_video_mode(self, mock_mp_image):
        """Verify processing in VIDEO mode calls detect_for_video."""
        mock_landmarker = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        timestamp = 1000
        
        result, mp_img = process_frame_with_mediapipe(
            frame, mock_landmarker, timestamp, use_video_mode=True
        )
        
        # Should create an mp.Image
        mock_mp_image.assert_called_once()
        # Should call detect_for_video
        mock_landmarker.detect_for_video.assert_called_once_with(ANY, timestamp)
        # Should return result
        assert result == mock_landmarker.detect_for_video.return_value

    def test_process_frame_image_mode(self, mock_mp_image):
        """Verify processing in IMAGE mode calls detect."""
        mock_landmarker = MagicMock()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, mp_img = process_frame_with_mediapipe(
            frame, mock_landmarker, timestamp_ms=0, use_video_mode=False
        )
        
        mock_landmarker.detect.assert_called_once()
        mock_landmarker.detect_for_video.assert_not_called()

    def test_process_frame_exception_handling(self, mock_mp_image):
        """Verify exceptions during inference return None."""
        mock_landmarker = MagicMock()
        mock_landmarker.detect_for_video.side_effect = RuntimeError("MediaPipe Crash")
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        result, _ = process_frame_with_mediapipe(frame, mock_landmarker, 0)
        
        assert result is None

    # --- DRAWING TESTS ---

    def test_draw_hand_landmarks_logic(self):
        """Verify drawing functions (cv2.line, cv2.circle) are called."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create mock landmarks (Wrist + Index Finger)
        mock_wrist = MagicMock(x=0.5, y=0.5)
        mock_index = MagicMock(x=0.6, y=0.6)
        
        # List of lists (representing multiple hands)
        landmarks = [[mock_wrist, mock_index]]
        
        # Connection from 0 to 1
        connections = [(0, 1)]
        
        with patch("cv2.line") as mock_line, \
             patch("cv2.circle") as mock_circle:
             
            draw_hand_landmarks(frame, landmarks, connections)
            
            # Should draw 1 line (Wrist -> Index)
            mock_line.assert_called_once()
            
            # Should draw 1 circle (Wrist only, per the code logic)
            mock_circle.assert_called_once()

def run_tests_directly() -> None:
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()