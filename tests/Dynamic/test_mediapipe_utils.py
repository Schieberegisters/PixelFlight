import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from Dynamic import mediapipe_utils
from Dynamic.mediapipe_utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    annotate_frame
)

class TestMediaPipeUtils:

    @pytest.fixture
    def mock_image(self) -> np.ndarray:
        """Creates a dummy NumPy BGR image."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        """Mocks the MediaPipe Holistic model."""
        model = MagicMock()
        model.process.return_value = MagicMock(name="results")
        return model

    def test_mediapipe_detection_flow(self, mock_image: np.ndarray, mock_model: MagicMock) -> None:
        """Verifies color conversion logic and model processing flow."""
        with patch("cv2.cvtColor", side_effect=lambda img, code: img) as mock_cvt:
            
            image, results = mediapipe_detection(mock_image, mock_model)
            
            # Must call BGR2RGB before processing and RGB2BGR after
            assert mock_cvt.call_count == 2
            mock_model.process.assert_called_once()
            assert results == mock_model.process.return_value

    def test_draw_styled_landmarks_calls(self, mock_image: np.ndarray) -> None:
        """Ensures all 4 components (Face, Pose, Hands) are drawn."""
        mock_results = MagicMock()
        
        with patch.object(mediapipe_utils, "mp_drawing") as mock_mp_draw:
            draw_styled_landmarks(mock_image, mock_results)
            
            # Expecting separate calls for Face Mesh, Pose, and both hands
            assert mock_mp_draw.draw_landmarks.call_count == 4
            
            calls = mock_mp_draw.draw_landmarks.call_args_list
            assert calls[0][0][1] == mock_results.face_landmarks
            assert calls[1][0][1] == mock_results.pose_landmarks
            assert calls[2][0][1] == mock_results.left_hand_landmarks
            assert calls[3][0][1] == mock_results.right_hand_landmarks

    def test_annotate_frame_styled(self, mock_image: np.ndarray, mock_model: MagicMock) -> None:
        """Verifies full integration using custom styling."""
        with patch.object(mediapipe_utils, "mediapipe_detection") as mock_detect, \
             patch.object(mediapipe_utils, "draw_styled_landmarks") as mock_draw_styled:
            
            mock_results = MagicMock()
            mock_detect.return_value = (mock_image, mock_results)
            
            out_img, out_res = annotate_frame(mock_image, mock_model, styled=True)
            
            mock_detect.assert_called_once_with(mock_image, mock_model)
            mock_draw_styled.assert_called_once_with(mock_image, mock_results)
            
            assert out_img is mock_image
            assert out_res is mock_results

    def test_annotate_frame_unstyled(self, mock_image: np.ndarray, mock_model: MagicMock) -> None:
        """Verifies fallback to default MediaPipe styling."""
        with patch.object(mediapipe_utils, "mediapipe_detection") as mock_detect, \
             patch.object(mediapipe_utils, "draw_styled_landmarks") as mock_draw_styled, \
             patch.object(mediapipe_utils, "mp_drawing") as mock_mp_draw_default:
            
            mock_results = MagicMock()
            mock_detect.return_value = (mock_image, mock_results)
            
            annotate_frame(mock_image, mock_model, styled=False)
            
            # Styled drawing should be skipped in favor of default
            mock_draw_styled.assert_not_called()
            assert mock_mp_draw_default.draw_landmarks.call_count == 4

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