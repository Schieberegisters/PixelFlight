import numpy as np
from typing import Tuple, Any, Final, List
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- LOCAL MODULES ---
from Dynamic.data_preprocessing import preprocess_data
from Dynamic.model import load_model
from config.dynamic import MODEL_PATH as DEFAULT_MODEL_PATH



def evaluate_model(model_path: str = DEFAULT_MODEL_PATH) -> Tuple[float, np.ndarray]:
    """
    Evaluates the trained model on the test dataset split.
    
    Returns:
        A tuple containing the accuracy score and the multilabel confusion matrix.
    """
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path)
    
    # Preprocess data (loading only the test split)
    print("Fetching test dataset...")
    _, x_test, _, y_test = preprocess_data()
    
    print(f"Predicting on {len(x_test)} samples...")
    predictions = model.predict(x_test)
    
    # Convert one-hot encoded labels to class indices
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate performance metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    conf_matrix = multilabel_confusion_matrix(y_true, y_pred)
    
    _print_results(accuracy, conf_matrix)
    
    return accuracy, conf_matrix


def _print_results(accuracy: float, conf_matrix: np.ndarray) -> None:
    """Internal helper to format evaluation output."""
    print("-" * 30)
    print(f"EVALUATION RESULTS")
    print("-" * 30)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix (Multilabel):\n{conf_matrix}")
    print("-" * 30)


if __name__ == "__main__":
    evaluate_model()