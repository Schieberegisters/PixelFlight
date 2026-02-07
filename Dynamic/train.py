"""Training entrypoint with integrated evaluation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from .data_preprocessing import preprocess_data
from .model import create_model
from config.dynamic import EPOCHS, LOG_DIR, MODEL_PATH
from config.gestures import DYNAMIC_ACTIONS


def train_and_evaluate_model() -> Any:
    """Train the model and print a quick evaluation on the held-out split."""

    print("[INFO] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()
    
    print(f"[INFO] Training data shape: {X_train.shape}")
    print(f"[INFO] Test data shape: {X_test.shape}")
    
    print("[INFO] Creating model...")
    model = create_model()
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    tb_callback = TensorBoard(log_dir=LOG_DIR)
    
    early_stopping = EarlyStopping(
        monitor="val_categorical_accuracy",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )
    
    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_categorical_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
    
    print("[INFO] Starting training...")
    _history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=32,
        callbacks=[tb_callback, early_stopping, checkpoint],
    )
    
    print(f"[INFO] Saving final model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    print("\n" + "="*50)
    print("STARTING EVALUATION (on already loaded X_test)")
    print("="*50)
    
    yhat = model.predict(X_test)
    
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    
    accuracy = accuracy_score(ytrue, yhat)
    confusion_matrix = multilabel_confusion_matrix(ytrue, yhat)
    
    print(f"\n[RESULT] Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n[RESULT] Detailed Classification Report:")
    print(classification_report(ytrue, yhat, target_names=DYNAMIC_ACTIONS))
    
    print("\n[RESULT] Confusion Matrix (Raw):")
    print(confusion_matrix)
    
    print("\n--- Confusion Analysis ---")
    for i, action in enumerate(DYNAMIC_ACTIONS):
        cm_class = confusion_matrix[i]
        tp = cm_class[1, 1]
        fn = cm_class[1, 0]
        fp = cm_class[0, 1]
        
        print(f"{action.ljust(15)} -> Corretti: {tp}, Persi: {fn}, Inventati: {fp}")

    print("="*50)
    print("Training and Evaluation Completed Successfully.")
    
    return model

if __name__ == "__main__":
    train_and_evaluate_model()