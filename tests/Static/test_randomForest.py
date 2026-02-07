import sys
import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
# Allows importing the Static module if the test is run directly
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

from Static.randomForest import (
    load_and_prepare_data,
    train_classifier,
    evaluate_model,
    persist_model
)

class TestRandomForestTraining:

    @pytest.fixture
    def synthetic_csv(self, tmp_path):
        """Creates a synthetic CSV file for loading tests."""
        csv_path = tmp_path / "test_gestures.csv"
        data = {
            'feat1': np.random.rand(10),
            'feat2': np.random.rand(10),
            'gesture': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    # --- DATA PREPARATION TESTS ---

    def test_load_and_prepare_data_split(self, synthetic_csv):
        """Verify the dataset is split correctly and stratified."""
        label_col = 'gesture'
        x_train, x_test, y_train, y_test = load_and_prepare_data(synthetic_csv, label_col)

        # Verify proportions (test_size=0.2 of 10 rows = 2 rows)
        assert len(x_test) == 2
        assert len(x_train) == 8
        
        # Verify stratification (class distribution should be balanced)
        # Expected 1 'A' and 1 'B' in the test set
        assert (y_test == 'A').sum() == 1
        assert (y_test == 'B').sum() == 1

    # --- TRAINING TESTS ---

    def test_train_classifier_logic(self):
        """Verify the classifier is trained and returned."""
        x_train = pd.DataFrame({'f1': [1, 2, 3, 4], 'f2': [4, 3, 2, 1]})
        y_train = pd.Series(['A', 'A', 'B', 'B'])

        model = train_classifier(x_train, y_train)

        # Ensure it is a RandomForest instance
        assert hasattr(model, "predict")
        assert model.n_estimators == 100
        
        # Ensure the model is "fitted" (has classes_ attribute)
        assert hasattr(model, "classes_")

    # --- EVALUATION TESTS ---

    def test_evaluate_model_output(self, capsys):
        """Verify the evaluation function prints expected results."""
        model = MagicMock()
        model.predict.return_value = np.array(['A', 'B'])
        x_test = pd.DataFrame({'f1': [1, 2]})
        y_test = pd.Series(['A', 'B'])

        evaluate_model(model, x_test, y_test)

        captured = capsys.readouterr()
        assert "Test Accuracy: 100.00%" in captured.out
        assert "Detailed Classification Report" in captured.out

    # --- PERSISTENCE TESTS ---

    def test_persist_model_io(self, tmp_path):
        """Verify model saving to disk works for serializable objects."""
        # FIX: Using a simple dict. MagicMock cannot be pickled/serialized by joblib.
        model = {"dummy_param": 123, "type": "test_model"}
        save_path = tmp_path / "models" / "test_model.joblib"
        
        persist_model(model, str(save_path))

        # Verify file creation
        assert save_path.exists()
        
        # Verify the file is a loadable joblib file and contains the data
        loaded_obj = joblib.load(str(save_path))
        assert loaded_obj == model

    @patch("Static.randomForest.joblib.dump")
    def test_persist_model_error_handling(self, mock_dump, capsys):
        """Verify error handling during model saving failures."""
        mock_dump.side_effect = OSError("Disk Full")
        model = MagicMock()
        
        persist_model(model, "dummy_path.joblib")
        
        captured = capsys.readouterr()
        assert "Failed to save model: Disk Full" in captured.out

def run_tests_directly() -> None:
    """Entry point for direct script execution."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()