"""
Trains a RandomForestClassifier for static hand gestures.
Data is loaded from CSV, evaluated with a stratified split, and persisted via joblib.
"""

from __future__ import annotations
from pathlib import Path
from typing import Final, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR: Final[Path] = Path(__file__).resolve().parent

# Dataset Paths
CSV_FILE_PATH: Final[str] = str(BASE_DIR / "gesture_training_data_expanded.csv")
LABEL_COLUMN: Final[str] = 'gesture'

# Persistence Settings
MODEL_OUTPUT_PATH: Final[str] = str(BASE_DIR / "models" / "140k.joblib")


def load_and_prepare_data(
    file_path: str, 
    label_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads CSV and performs a stratified train/test split."""
    df = pd.read_csv(file_path)
    
    x = df.drop(columns=[label_col])
    y = df[label_col]

    # Stratify ensures the same class distribution in train and test sets
    return train_test_split(
        x, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )


def train_classifier(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Initializes and trains the RandomForest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Prints performance metrics for the trained model."""
    predictions = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))


def persist_model(model: RandomForestClassifier, save_path: str) -> None:
    """Saves the model artifact to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        joblib.dump(model, save_path)
        print(f"\nModel successfully persisted to: {save_path}")
    except OSError as e:
        print(f"\nFailed to save model: {e}")


def main() -> None:
    """Main execution pipeline."""
    # Data Preparation
    x_train, x_test, y_train, y_test = load_and_prepare_data(CSV_FILE_PATH, LABEL_COLUMN)

    # Training
    gesture_model = train_classifier(x_train, y_train)

    # Evaluation
    evaluate_model(gesture_model, x_test, y_test)

    # Persistence
    persist_model(gesture_model, MODEL_OUTPUT_PATH)
if __name__ == "__main__":
    main()