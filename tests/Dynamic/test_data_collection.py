import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import module object for patching
from Dynamic import data_collection

# Import specific components for testing
from Dynamic.data_collection import (
    DataAugmenter, 
    DataCollector, 
    POSE_LEN, 
    FACE_LEN, 
    HAND_LEN
)

class TestDataCollection:

    # --- DATA AUGMENTER TESTS (MATH LOGIC) ---

    def test_get_centroid_simple(self):
        """Verify centroid calculation ignores zeros and finds mean."""
        # Use non-zero values because 0.0 is treated as 'undetected'
        data = np.zeros(POSE_LEN + FACE_LEN + HAND_LEN * 2)
        
        # Point 1 at (0.2, 0.2)
        data[0] = 0.2; data[1] = 0.2 
        # Point 2 at (0.8, 0.8)
        data[4] = 0.8; data[5] = 0.8 
        
        # Centroid should be (0.5, 0.5)
        cx, cy = DataAugmenter.get_centroid(data)
        assert cx == 0.5
        assert cy == 0.5

    def test_flip_keypoints_logic(self):
        """Verify X-flip (1-x) and anatomical swapping."""
        data = np.zeros(1662) # Total len
        
        # Set Left Eye (Index 1 -> data index 4)
        data[4] = 0.2
        # Set Right Eye (Index 4 -> data index 16)
        data[16] = 0.8
        
        flipped = DataAugmenter.flip_keypoints(data)
        
        # Original Left (0.2) flips to 0.8 and moves to Right slot
        assert np.isclose(flipped[16], 0.8)
        
        # Original Right (0.8) flips to 0.2 and moves to Left slot
        assert np.isclose(flipped[4], 0.2) 

    def test_apply_rotation_90_degrees(self):
        """Verify 2D rotation logic."""
        data = np.zeros(1662)
        # Point 1: (0.6, 0.5)
        data[0] = 0.6; data[1] = 0.5
        # Point 2: (0.4, 0.5) 
        data[4] = 0.4; data[5] = 0.5
        
        # Centroid is (0.5, 0.5)
        # Point 1 relative to C: (0.1, 0)
        # Rotate 90 deg -> (0, 0.1) relative to C
        # Absolute New Pos -> (0.5, 0.6)
        
        rotated = DataAugmenter.apply_rotation(data, 90)
        
        assert np.isclose(rotated[0], 0.5, atol=1e-7)
        assert np.isclose(rotated[1], 0.6, atol=1e-7)

    def test_apply_translation(self):
        """Verify simple X/Y offset."""
        data = np.zeros(1662)
        data[0] = 0.5; data[1] = 0.5
        
        translated = DataAugmenter.apply_translation(data, 0.1, -0.1)
        
        assert np.isclose(translated[0], 0.6)
        assert np.isclose(translated[1], 0.4)

    # --- DATA COLLECTOR TESTS (WORKFLOW & IO) ---

    def test_setup_directories_rewrite(self):
        """Verify it deletes old data if rewrite=True."""
        collector = DataCollector(rewrite=True)
        
        with patch("os.path.exists", return_value=True), \
             patch("shutil.rmtree") as mock_rm, \
             patch("os.makedirs") as mock_mk:
            
            collector._setup_directories()
            
            mock_rm.assert_called_once()
            assert mock_mk.call_count >= 1

    def test_generate_augmentations_execution(self):
        """Verify it creates 3 augmented versions (rot, trans, scale)."""
        collector = DataCollector()
        
        # Define logic for path checking: 
        # - Augmentation folders (don't exist yet -> return False)
        # - Source files (exist -> return True)
        def exists_side_effect(path):
            if "aug" in str(path): return False
            return True

        with patch("os.path.exists", side_effect=exists_side_effect), \
             patch("os.makedirs"), \
             patch("os.listdir", return_value=[]), \
             patch("numpy.load", return_value=np.zeros(1662)), \
             patch("numpy.save") as mock_save:
            
            with patch("Dynamic.data_collection.SEQUENCE_LENGTH", 1): 
                collector._generate_augmentations("test_action", 0, False)
                
                # Should save 3 times (rot, trans, scale)
                assert mock_save.call_count == 3

    def test_extend_dataset_logic(self):
        """Test the logic that finds max index and creates new copies."""
        collector = DataCollector(append=True)
        
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=["0", "1", "2"]), \
             patch("os.makedirs") as mock_mk, \
             patch("numpy.load", return_value=np.zeros(1662)), \
             patch("numpy.save") as mock_save, \
             patch("random.choice", return_value=0), \
             patch.object(collector, "_generate_augmentations") as mock_aug:
            
            with patch("Dynamic.data_collection.NO_SEQUENCES", 1), \
                 patch("Dynamic.data_collection.SEQUENCE_LENGTH", 1):
                
                collector._extend_dataset()
                
                assert mock_mk.called
                assert mock_save.called
                assert mock_aug.called

    def test_run_main_loop_flow(self):
        """Test the main run loop without opening a real camera."""
        collector = DataCollector()
        
        mock_cap = MagicMock()
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        collector.cap = mock_cap
        
        # MOCK EVERYTHING to prevent real IO and UI
        with patch.object(collector, "_setup_directories"), \
             patch.object(collector, "_show_countdown", return_value=True), \
             patch("Dynamic.data_collection.mediapipe_detection") as mock_detect, \
             patch("Dynamic.data_collection.extract_keypoints") as mock_extract, \
             patch("Dynamic.data_collection.draw_styled_landmarks") as mock_draw, \
             patch("numpy.save"), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=0), \
             patch("os.makedirs"), \
             patch("os.path.exists", return_value=False): # <--- FIX: Force "Not Exists" to trigger recording
            
            mock_detect.return_value = (np.zeros((100,100,3)), MagicMock())
            
            with patch("Dynamic.data_collection.ACTIONS", ["TEST_ACTION"]), \
                 patch("Dynamic.data_collection.NO_SEQUENCES", 1), \
                 patch("Dynamic.data_collection.SEQUENCE_LENGTH", 1):
                 
                 collector.run()
                 
                 # Now this assertion should pass because we didn't skip the loop
                 mock_detect.assert_called()
                 mock_extract.assert_called()
                 mock_draw.assert_called()

def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __name__ if __name__ != "__main__" else __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()