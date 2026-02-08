"""
Model definition for sign language recognition.
Defines the LSTM neural network architecture.
"""
from __future__ import annotations
import sys
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Final
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.dynamic import INPUT_SHAPE, LOSS, METRICS, MODEL_PATH
from config.gestures import DYNAMIC_ACTIONS

# Legacy model hyperparameters (kept local to avoid polluting global config).
LSTM_UNITS: Final[tuple[int, int, int]] = (64, 128, 64)
DENSE_UNITS: Final[tuple[int, int]] = (64, 32)


def create_model() -> Sequential:
    """Create and compile the LSTM model."""

    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        LSTM_UNITS[0],
        return_sequences=True,
        activation='relu',
        input_shape=INPUT_SHAPE,
    ))
    
    # Second LSTM layer
    model.add(LSTM(
        LSTM_UNITS[1],
        return_sequences=True,
        activation='relu'
    ))
    
    # Third LSTM layer
    model.add(LSTM(
        LSTM_UNITS[2],
        return_sequences=False,
        activation='relu'
    ))
    
    # Dense layers
    model.add(Dense(DENSE_UNITS[0], activation='relu'))
    model.add(Dense(DENSE_UNITS[1], activation='relu'))
    
    # Output layer
    model.add(Dense(DYNAMIC_ACTIONS.shape[0], activation='softmax'))
    
    # Compile model
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(learning_rate=0.000005)
    model.compile(
        optimizer=optimizer,
        loss=LOSS,
        metrics=METRICS,
    )
    
    return model


def load_model(model_path: str | None = None) -> Sequential:
    """Load a saved model from disk."""

    if model_path is None:
        model_path = MODEL_PATH
    
    from tensorflow.keras.models import load_model as keras_load_model
    try:
        return keras_load_model(model_path)
    except Exception as e:
        # If loading full model fails, try loading weights only
        model = create_model()
        model.load_weights(model_path)
        return model

