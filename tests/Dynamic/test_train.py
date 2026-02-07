import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import module object for patching
from Dynamic import train
from Dynamic.train import train_and_evaluate_model

class TestTrainLoop:

    @pytest.fixture
    def mock_training_dependencies(self):
        """Mocks all external dependencies: Data, Model, Callbacks, OS."""
        with patch("Dynamic.train.preprocess_data") as mock_data, \
             patch("Dynamic.train.create_model") as mock_create_model, \
             patch("Dynamic.train.TensorBoard") as mock_tb, \
             patch("Dynamic.train.EarlyStopping") as mock_es, \
             patch("Dynamic.train.ModelCheckpoint") as mock_ckpt, \
             patch("os.makedirs") as mock_makedirs, \
             patch("Dynamic.train.classification_report") as mock_report, \
             patch("Dynamic.train.DYNAMIC_ACTIONS", ["ACTION_1", "ACTION_2"]): # <--- FIX 1: Sync actions with data
            
            # 1. Setup Mock Data
            # X_train: (10 samples, 20 frames, 1662 feats)
            X_train = np.zeros((10, 20, 1662))
            X_test  = np.zeros((2, 20, 1662))
            
            # Y one-hot: 2 classes (matching the patched DYNAMIC_ACTIONS len)
            y_train = np.array([[1, 0]] * 10)
            y_test  = np.array([[1, 0], [0, 1]]) # Class 0, Class 1
            
            mock_data.return_value = (X_train, X_test, y_train, y_test)
            
            # 2. Setup Mock Model
            mock_model = MagicMock()
            mock_model.fit.return_value = MagicMock(history={'accuracy': [0.9]})
            
            # Predictions must match shape (N_samples, N_classes) -> (2, 2)
            mock_model.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
            
            mock_create_model.return_value = mock_model
            mock_report.return_value = "Mock Classification Report"

            yield {
                "preprocess": mock_data,
                "create_model": mock_create_model,
                "model": mock_model,
                "tb": mock_tb,
                "es": mock_es,
                "ckpt": mock_ckpt,
                "makedirs": mock_makedirs,
                "X_test": X_test,
                "y_test": y_test
            }

    def test_train_and_evaluate_execution_flow(self, mock_training_dependencies):
        """Verifies the complete pipeline flow."""
        
        # Execution
        model = train_and_evaluate_model()
        
        deps = mock_training_dependencies
        
        # 1. Data Loading
        deps["preprocess"].assert_called_once()
        
        # 2. Model Creation
        deps["create_model"].assert_called_once()
        
        # 3. Log Directory
        deps["makedirs"].assert_called_with(train.LOG_DIR, exist_ok=True)
        
        # 4. Callbacks Init
        deps["tb"].assert_called_once()
        deps["es"].assert_called_once()
        deps["ckpt"].assert_called_once()
        
        # 5. Fit
        deps["model"].fit.assert_called_once_with(
            ANY, ANY, 
            validation_data=(deps["X_test"], deps["y_test"]),
            epochs=train.EPOCHS,
            batch_size=32,
            callbacks=ANY
        )
        
        # 6. Save
        deps["model"].save.assert_called_once_with(train.MODEL_PATH)
        
        # 7. Predict/Eval
        deps["model"].predict.assert_called_once_with(deps["X_test"])
        
        assert model == deps["model"]

    def test_evaluation_metric_calculation(self, mock_training_dependencies, capsys):
        """Verifies console output contains accuracy and confusion matrix info."""
        
        train_and_evaluate_model()
        
        captured = capsys.readouterr()
        
        # Check for key strings in output
        assert "Final Accuracy: 1.0000" in captured.out
        assert "STARTING EVALUATION" in captured.out
        assert "Confusion Matrix" in captured.out
        assert "Mock Classification Report" in captured.out
        
        # Check that our specific patched action names appear
        assert "ACTION_1" in captured.out

    def test_callback_configuration(self, mock_training_dependencies):
        """Verifies callbacks are configured with correct monitor metric."""
        # FIX 2: We must run the function to trigger the mock calls
        train_and_evaluate_model()
        
        deps = mock_training_dependencies
        
        # Early Stopping check
        _, es_kwargs = deps["es"].call_args
        assert es_kwargs["monitor"] == "val_categorical_accuracy"
        assert es_kwargs["restore_best_weights"] is True
        
        # Model Checkpoint check
        _, ckpt_kwargs = deps["ckpt"].call_args
        assert ckpt_kwargs["monitor"] == "val_categorical_accuracy"
        assert ckpt_kwargs["save_best_only"] is True

def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()