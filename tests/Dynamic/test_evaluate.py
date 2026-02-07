import sys
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from Dynamic.evaluate import evaluate_model

class TestEvaluate:
    @pytest.fixture
    def mock_evaluate_deps(self) -> Generator[dict[str, Any], None, None]:
        """Mocks heavy dependencies: model loading and data preprocessing."""
        with patch("Dynamic.evaluate.load_model") as mock_load_model, \
             patch("Dynamic.evaluate.preprocess_data") as mock_preprocess:
            
            model_instance = MagicMock()
            mock_load_model.return_value = model_instance
            
            # Mock data setup: 2 samples, 20 frames, 1662 features
            x_test = np.zeros((2, 20, 1662))
            # One-hot labels: [1, 0] (Class 0) and [0, 1] (Class 1)
            y_test = np.array([[1, 0], [0, 1]])
            
            mock_preprocess.return_value = (None, x_test, None, y_test)
            
            yield {
                "model": model_instance,
                "load_model": mock_load_model,
                "preprocess": mock_preprocess,
                "x_test": x_test,
                "y_test": y_test
            }

    def test_evaluate_model_full_flow(self, mock_evaluate_deps: dict[str, Any]) -> None:
        """Verifies full evaluation flow: loading, prediction, and metrics calculation."""
        # Simulate 100% accuracy: probabilities [0.9, 0.1] and [0.2, 0.8]
        mock_predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_evaluate_deps["model"].predict.return_value = mock_predictions

        accuracy, conf_matrix = evaluate_model("fake_path.keras")

        mock_evaluate_deps["load_model"].assert_called_once_with("fake_path.keras")
        mock_evaluate_deps["preprocess"].assert_called_once()
        mock_evaluate_deps["model"].predict.assert_called_once_with(mock_evaluate_deps["x_test"])
        
        assert accuracy == 1.0
        assert isinstance(conf_matrix, np.ndarray)
        # Multilabel matrix for 2 classes has shape (2, 2, 2)
        assert conf_matrix.shape == (2, 2, 2)

    def test_evaluate_model_half_accuracy(self, mock_evaluate_deps: dict[str, Any]) -> None:
        """Verifies metrics calculation with 50% accuracy."""
        # Simulate one correct and one incorrect prediction
        mock_predictions = np.array([[0.9, 0.1], [0.7, 0.3]])
        mock_evaluate_deps["model"].predict.return_value = mock_predictions

        accuracy, _ = evaluate_model()

        assert accuracy == 0.5

    def test_print_results_output(self, capsys: pytest.CaptureFixture) -> None:
        """Verifies internal helper console output."""
        from Dynamic.evaluate import _print_results
        
        test_acc = 0.9555
        test_matrix = np.array([[[1, 0], [0, 1]]])
        
        _print_results(test_acc, test_matrix)
        
        captured = capsys.readouterr()
        assert "EVALUATION RESULTS" in captured.out
        assert "0.9555" in captured.out
        assert "Confusion Matrix" in captured.out

def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()