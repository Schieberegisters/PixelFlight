import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from Dynamic.lstm import create_model, load_model, LSTM_UNITS, DENSE_UNITS

class TestLSTMModel:
    def test_create_model_architecture(self) -> None:
        """Verifies architecture: 3 LSTM layers, 2 Dense layers, 1 Output layer."""
        model = create_model()
        
        # Total expected layers added via .add()
        assert len(model.layers) == 6
        
        # LSTM Layer 1
        assert model.layers[0].units == LSTM_UNITS[0]
        assert model.layers[0].return_sequences is True
        
        # LSTM Layer 2
        assert model.layers[1].units == LSTM_UNITS[1]
        assert model.layers[1].return_sequences is True
        
        # LSTM Layer 3 (Final sequence processing)
        assert model.layers[2].units == LSTM_UNITS[2]
        assert model.layers[2].return_sequences is False
        
        # Dense Layers
        assert model.layers[3].units == DENSE_UNITS[0]
        assert model.layers[4].units == DENSE_UNITS[1]
        
        # Output Layer (Softmax classification)
        from config.gestures import DYNAMIC_ACTIONS
        assert model.layers[5].units == len(DYNAMIC_ACTIONS)
        assert model.layers[5].activation.__name__ == 'softmax'

    def test_model_input_shape(self) -> None:
        """Ensures the model accepts the configured input shape."""
        from config.dynamic import INPUT_SHAPE
        model = create_model()
        
        # input_shape returns (None, frames, features)
        assert model.input_shape[1:] == INPUT_SHAPE

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_success(self, mock_keras_load: MagicMock) -> None:
        """Verifies standard model loading via Keras API."""
        mock_model = MagicMock()
        mock_keras_load.return_value = mock_model
        
        loaded = load_model("fake_path.keras")
        
        mock_keras_load.assert_called_once_with("fake_path.keras")
        assert loaded == mock_model

    @patch("tensorflow.keras.models.load_model", side_effect=Exception("Load failed"))
    @patch("Dynamic.lstm.create_model")
    def test_load_model_fallback_weights(self, mock_create: MagicMock, mock_keras_load: MagicMock) -> None:
        """Verifies weights-only fallback if full model loading fails."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        loaded = load_model("fake_path.h5")
        
        # Should initialize new architecture and apply weights
        mock_create.assert_called_once()
        mock_model.load_weights.assert_called_once_with("fake_path.h5")
        assert loaded == mock_model

    def test_model_compilation(self) -> None:
        """Validates compilation parameters including optimizer and learning rate."""
        from config.dynamic import LOSS
        model = create_model()
        
        # Loss function check (handles Keras loss object wrappers)
        assert LOSS in str(model.loss).lower() or model.loss == LOSS
        
        assert model.optimizer is not None
        # Learning rate validation using approx for float precision
        lr = model.optimizer.learning_rate.numpy()
        assert float(lr) == pytest.approx(0.000005)

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