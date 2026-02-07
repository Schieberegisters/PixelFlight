import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import del modulo oggetto
from Dynamic import lstm
from Dynamic.lstm import create_model, load_model, LSTM_UNITS, DENSE_UNITS

class TestLSTMModel:

    def test_create_model_architecture(self):
        """
        Verifica che l'architettura del modello corrisponda alle specifiche:
        3 strati LSTM + 2 strati Dense + 1 strato Output.
        """
        model = create_model()
        
        # Verifica numero totale di strati (6 strati aggiunti via .add())
        assert len(model.layers) == 6
        
        # Verifica gli strati LSTM
        # Strato 1: return_sequences=True
        assert model.layers[0].units == LSTM_UNITS[0]
        assert model.layers[0].return_sequences is True
        
        # Strato 2: return_sequences=True
        assert model.layers[1].units == LSTM_UNITS[1]
        assert model.layers[1].return_sequences is True
        
        # Strato 3: return_sequences=False
        assert model.layers[2].units == LSTM_UNITS[2]
        assert model.layers[2].return_sequences is False
        
        # Verifica strati Dense
        assert model.layers[3].units == DENSE_UNITS[0]
        assert model.layers[4].units == DENSE_UNITS[1]
        
        # Verifica strato di Output (Softmax)
        from config.gestures import DYNAMIC_ACTIONS
        assert model.layers[5].units == len(DYNAMIC_ACTIONS)
        assert model.layers[5].activation.__name__ == 'softmax'

    def test_model_input_shape(self):
        """Verifica che il modello accetti l'input shape configurato."""
        from config.dynamic import INPUT_SHAPE
        model = create_model()
        
        # model.input_shape restituisce (None, frames, features)
        assert model.input_shape[1:] == INPUT_SHAPE

    @patch("tensorflow.keras.models.load_model")
    def test_load_model_success(self, mock_keras_load):
        """Verifica il caricamento standard del modello via Keras."""
        mock_model = MagicMock()
        mock_keras_load.return_value = mock_model
        
        loaded = load_model("fake_path.keras")
        
        mock_keras_load.assert_called_once_with("fake_path.keras")
        assert loaded == mock_model

    @patch("tensorflow.keras.models.load_model", side_effect=Exception("Load failed"))
    @patch("Dynamic.lstm.create_model")
    def test_load_model_fallback_weights(self, mock_create, mock_keras_load):
        """Verifica il fallback: se il caricamento fallisce, prova a caricare solo i pesi."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        # Esecuzione
        loaded = load_model("fake_path.h5")
        
        # Dovrebbe aver creato un nuovo modello e chiamato load_weights
        mock_create.assert_called_once()
        mock_model.load_weights.assert_called_once_with("fake_path.h5")
        assert loaded == mock_model

    def test_model_compilation(self):
        """Verifica che il modello sia compilato con l'ottimizzatore e le metriche corrette."""
        from config.dynamic import LOSS
        model = create_model()
        
        # Verifica loss function
        # Nota: Keras può avvolgere la stringa in un oggetto loss
        assert LOSS in str(model.loss).lower() or model.loss == LOSS
        
        # Verifica presenza optimizer
        assert model.optimizer is not None
        # Verifica learning rate (vicino a 0.000005)
        lr = model.optimizer.learning_rate.numpy()
        assert float(lr) == pytest.approx(0.000005)

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()