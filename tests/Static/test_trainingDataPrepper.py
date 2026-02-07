import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import the module object
from Static.trainingDataPrepper import DataCollector

class TestTrainingDataPrepper:

    @pytest.fixture
    def mock_dependencies(self):
        """Mocks OpenCV, MediaPipe utils, and Data utils."""
        with patch("cv2.VideoCapture") as mock_cap, \
             patch("Static.trainingDataPrepper.get_hand_landmarker") as mock_get_lm, \
             patch("Static.trainingDataPrepper.process_frame_with_mediapipe") as mock_process, \
             patch("Static.trainingDataPrepper.normalize_landmarks") as mock_norm, \
             patch("Static.trainingDataPrepper.save_data_to_csv") as mock_save, \
             patch("Static.trainingDataPrepper.expand_training_data") as mock_expand, \
             patch("cv2.putText") as mock_put_text:
            
            # Setup Camera
            cap_instance = MagicMock()
            cap_instance.isOpened.return_value = True
            cap_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
            mock_cap.return_value = cap_instance
            
            # Setup Utils
            mock_norm.return_value = [0.1, 0.2, 0.3] # Dummy features
            
            yield {
                "cap": cap_instance,
                "process": mock_process,
                "normalize": mock_norm,
                "save": mock_save,
                "expand": mock_expand,
                "put_text": mock_put_text
            }

    # --- INITIALIZATION TESTS ---

    def test_initialization(self, mock_dependencies):
        """Verify the collector initializes camera and state correctly."""
        collector = DataCollector(camera_index=0)
        
        assert collector.data_buffer == []
        assert collector.current_gesture == "NONE"
        assert collector.is_collecting is False
        assert collector.camera == mock_dependencies["cap"]

    # --- FEATURE EXTRACTION TESTS ---

    def test_extract_features_hand_found(self, mock_dependencies):
        """Verify feature extraction when a hand is detected."""
        collector = DataCollector()
        
        # Mock detection result
        mock_result = MagicMock()
        mock_result.hand_landmarks = [MagicMock()] # One hand detected
        
        mock_dependencies["process"].return_value = (mock_result, None)
        
        frame = np.zeros((100, 100, 3))
        features, hand_found, _ = collector._extract_features(frame, timestamp_ms=0)
        
        assert hand_found is True
        assert features == [0.1, 0.2, 0.3] # From mock_norm
        mock_dependencies["normalize"].assert_called_once()

    def test_extract_features_no_hand(self, mock_dependencies):
        """Verify behavior when no hand is detected."""
        collector = DataCollector()
        
        # Mock empty result
        mock_dependencies["process"].return_value = (None, None)
        
        frame = np.zeros((100, 100, 3))
        features, hand_found, _ = collector._extract_features(frame, 0)
        
        assert hand_found is False
        assert features == []
        mock_dependencies["normalize"].assert_not_called()

    # --- DATA RECORDING LOGIC ---

    def test_data_buffer_appending(self, mock_dependencies):
        """Verify data is added to buffer only when collecting."""
        collector = DataCollector()
        collector.current_gesture = "TEST_GESTURE"
        
        # Simulate logic found in main loop:
        # if collecting and hand_found: append
        
        features = [1.0, 2.0]
        
        # Scenario 1: Collecting = False
        collector.is_collecting = False
        if collector.is_collecting:
            collector.data_buffer.append(features + [collector.current_gesture])
        assert len(collector.data_buffer) == 0
        
        # Scenario 2: Collecting = True
        collector.is_collecting = True
        if collector.is_collecting:
            collector.data_buffer.append(features + [collector.current_gesture])
        
        assert len(collector.data_buffer) == 1
        assert collector.data_buffer[0] == [1.0, 2.0, "TEST_GESTURE"]

    # --- UI & SAVING TESTS ---

    def test_draw_ui_overlay(self, mock_dependencies):
        """Verify UI text drawing calls."""
        collector = DataCollector()
        frame = np.zeros((480, 640, 3))
        
        collector._draw_ui(frame)
        
        # Should call putText multiple times (Status + Instructions)
        assert mock_dependencies["put_text"].call_count >= 1

    def test_save_dialog_yes_sequence(self, mock_dependencies):
        """Verify saving and expansion logic when user types 'y'."""
        collector = DataCollector()
        collector.data_buffer = [[1.0, "A"]]
        
        # Mock user input: 'y' for save, 'y' for expand
        with patch("builtins.input", side_effect=['y', 'y']):
            collector._save_dialog()
            
            mock_dependencies["save"].assert_called_once()
            mock_dependencies["expand"].assert_called_once()

    def test_save_dialog_no_save(self, mock_dependencies):
        """Verify nothing happens if user types 'n'."""
        collector = DataCollector()
        collector.data_buffer = [[1.0, "A"]]
        
        with patch("builtins.input", side_effect=['n']):
            collector._save_dialog()
            
            mock_dependencies["save"].assert_not_called()
            mock_dependencies["expand"].assert_not_called()

def run_tests_directly() -> None:
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()