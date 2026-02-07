import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    # Permette l'esecuzione diretta aggiungendo la root del progetto al path
    sys.path.append(str(Path(__file__).parents[2]))

# Import del modulo oggetto (per patch.object se necessario)
from Dynamic import evaluate
from Dynamic.evaluate import evaluate_model

class TestEvaluate:

    @pytest.fixture
    def mock_evaluate_deps(self):
        """Mock delle dipendenze pesanti: caricamento modello e caricamento dati."""
        with patch("Dynamic.evaluate.load_model") as mock_load_model, \
             patch("Dynamic.evaluate.preprocess_data") as mock_preprocess:
            
            # Setup del modello mock
            model_instance = MagicMock()
            mock_load_model.return_value = model_instance
            
            # Setup dei dati mock (X_train, X_test, y_train, y_test)
            # Creiamo 2 campioni di test
            x_test = np.zeros((2, 20, 1662))
            # Labels one-hot: [1, 0] (Classe 0) e [0, 1] (Classe 1)
            y_test = np.array([[1, 0], [0, 1]])
            
            mock_preprocess.return_value = (None, x_test, None, y_test)
            
            yield {
                "model": model_instance,
                "load_model": mock_load_model,
                "preprocess": mock_preprocess,
                "x_test": x_test,
                "y_test": y_test
            }

    def test_evaluate_model_full_flow(self, mock_evaluate_deps):
        """
        Verifica il flusso completo di valutazione:
        1. Caricamento modello e dati.
        2. Predizione.
        3. Calcolo accuratezza e matrice di confusione.
        """
        # Prepariamo le predizioni del modello mock
        # Simuliamo una predizione corretta al 100%
        # Il modello restituisce probabilità: [0.9, 0.1] e [0.2, 0.8]
        mock_predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
        mock_evaluate_deps["model"].predict.return_value = mock_predictions

        # Esecuzione
        accuracy, conf_matrix = evaluate_model("fake_path.keras")

        # ASSERTIONS
        
        # Verifica caricamento modello e dati
        mock_evaluate_deps["load_model"].assert_called_once_with("fake_path.keras")
        mock_evaluate_deps["preprocess"].assert_called_once()
        
        # Verifica che il modello abbia predetto sui dati corretti
        mock_evaluate_deps["model"].predict.assert_called_once_with(mock_evaluate_deps["x_test"])
        
        # Verifica risultati (accuratezza attesa: 1.0)
        assert accuracy == 1.0
        assert isinstance(conf_matrix, np.ndarray)
        # La matrice multilabel per 2 classi ha shape (2, 2, 2)
        assert conf_matrix.shape == (2, 2, 2)

    def test_evaluate_model_half_accuracy(self, mock_evaluate_deps):
        """Verifica il calcolo delle metriche con predizioni errate (50% accuracy)."""
        # y_true era [0, 1] (dagli indici di [[1,0], [0,1]])
        # Simuliamo y_pred = [0, 0] (la seconda è sbagliata)
        mock_predictions = np.array([[0.9, 0.1], [0.7, 0.3]])
        mock_evaluate_deps["model"].predict.return_value = mock_predictions

        accuracy, _ = evaluate_model()

        # 1 su 2 corretta = 0.5
        assert accuracy == 0.5

    def test_print_results_output(self, capsys):
        """Verifica che l'helper interno stampi correttamente i risultati a console."""
        from Dynamic.evaluate import _print_results
        
        test_acc = 0.9555
        test_matrix = np.array([[[1, 0], [0, 1]]])
        
        _print_results(test_acc, test_matrix)
        
        captured = capsys.readouterr()
        assert "EVALUATION RESULTS" in captured.out
        assert "0.9555" in captured.out
        assert "Confusion Matrix" in captured.out

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    # Nota: __name__ qui è "__main__", usiamo il file corrente
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()