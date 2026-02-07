import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Dynamic.visualization import render_probability_bars, START_Y, BAR_HEIGHT

class TestVisualization:

    @pytest.fixture
    def mock_frame(self) -> np.ndarray:
        """Creates a dummy black 640x480 BGR frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_config_colors(self) -> Generator[None, None, None]:
        """Mocks color configuration to avoid external dependency."""
        with patch("Dynamic.visualization.COLORS", [(255, 0, 0), (0, 255, 0)]):
            yield

    def test_render_bars_output_properties(self, mock_frame: np.ndarray, mock_config_colors: None) -> None:
        """Verifies frame validity and ensures the original is not modified in-place."""
        probabilities = [0.5, 0.8]
        actions = ["Act1", "Act2"]
        
        result = render_probability_bars(probabilities, actions, mock_frame)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == mock_frame.shape
        # Ensure deep copy behavior
        assert result is not mock_frame

    def test_drawing_calls_count(self, mock_frame: np.ndarray, mock_config_colors: None) -> None:
        """Verifies drawing operations match the number of provided actions."""
        probabilities = [0.1, 0.5, 0.9]
        actions = ["A", "B", "C"]
        
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probabilities, actions, mock_frame)
            
            assert mock_cv2.rectangle.call_count == 3
            assert mock_cv2.putText.call_count == 3

    def test_bar_geometry_logic(self, mock_frame: np.ndarray, mock_config_colors: None) -> None:
        """Validates Y-axis distribution of bars based on configured constants."""
        probabilities = [1.0, 1.0]
        actions = ["Top", "Bottom"]
        
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probabilities, actions, mock_frame)
            
            calls = mock_cv2.rectangle.call_args_list
            
            # Position for the first bar
            pt1_bar1 = calls[0][0][1] 
            y_top_1 = pt1_bar1[1]
            
            # Position for the second bar
            pt1_bar2 = calls[1][0][1]
            y_top_2 = pt1_bar2[1]
            
            assert y_top_1 == START_Y
            assert y_top_2 == START_Y + (1 * BAR_HEIGHT)
            assert y_top_2 > y_top_1

    def test_numpy_compatibility(self, mock_frame: np.ndarray, mock_config_colors: None) -> None:
        """Ensures function handles NumPy array inputs for probabilities and labels."""
        probs_np = np.array([0.3, 0.4])
        actions_np = np.array(["Np1", "Np2"])
        
        try:
            render_probability_bars(probs_np, actions_np, mock_frame)
        except Exception as e:
            pytest.fail(f"Function crashed with NumPy input: {e}")

    def test_color_cycling(self, mock_frame: np.ndarray) -> None:
        """Verifies modulo cycling of colors when actions outnumber provided colors."""
        custom_colors = [(255, 0, 0), (0, 0, 255)] 
        probabilities = [0.1, 0.2, 0.3]
        actions = ["1", "2", "3"]
        
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probabilities, actions, mock_frame, bar_colors=custom_colors)
            
            calls = mock_cv2.rectangle.call_args_list
            
            # Index 0 -> Color 0
            assert calls[0][0][3] == custom_colors[0]
            # Index 1 -> Color 1
            assert calls[1][0][3] == custom_colors[1]
            # Index 2 -> Cycle back to Color 0
            assert calls[2][0][3] == custom_colors[0]

def run_tests_directly() -> None:
    """Direct execution entry point for the test script."""
    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()