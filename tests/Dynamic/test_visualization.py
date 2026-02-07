import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import del modulo oggetto
from Dynamic import visualization
from Dynamic.visualization import render_probability_bars, START_Y, BAR_HEIGHT

class TestVisualization:

    @pytest.fixture
    def mock_frame(self):
        """Crea un frame nero 640x480."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_config_colors(self):
        """Mocka i colori per evitare dipendenze dal config reale."""
        with patch("Dynamic.visualization.COLORS", [(255, 0, 0), (0, 255, 0)]):
            yield

    def test_render_bars_output_properties(self, mock_frame, mock_config_colors):
        """Verifica che la funzione ritorni un frame valido e non modifichi l'originale in-place."""
        probs = [0.5, 0.8]
        actions = ["Act1", "Act2"]
        
        result = render_probability_bars(probs, actions, mock_frame)
        
        # Deve ritornare un array numpy
        assert isinstance(result, np.ndarray)
        # Stesse dimensioni
        assert result.shape == mock_frame.shape
        # Deve essere una copia (indirizzi di memoria diversi)
        assert result is not mock_frame

    def test_drawing_calls_count(self, mock_frame, mock_config_colors):
        """Verifica che vengano disegnate tante barre quante sono le azioni."""
        probs = [0.1, 0.5, 0.9] # 3 probabilità
        actions = ["A", "B", "C"] # 3 azioni
        
        # Patchiamo cv2 all'interno del modulo visualization
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probs, actions, mock_frame)
            
            # Ci aspettiamo 3 rettangoli e 3 testi
            assert mock_cv2.rectangle.call_count == 3
            assert mock_cv2.putText.call_count == 3

    def test_bar_geometry_logic(self, mock_frame, mock_config_colors):
        """Verifica che le barre siano posizionate correttamente sull'asse Y."""
        probs = [1.0, 1.0] # Lunghezza massima per semplicità
        actions = ["Top", "Bottom"]
        
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probs, actions, mock_frame)
            
            # Recuperiamo gli argomenti delle chiamate a rectangle
            calls = mock_cv2.rectangle.call_args_list
            
            # cv2.rectangle(img, pt1, pt2, color, ...)
            # pt1 è una tupla (x, y_top)
            
            # Prima barra (Indice 0)
            args_1 = calls[0][0] # argomenti posizionali
            pt1_bar1 = args_1[1] # (0, y_top)
            y_top_1 = pt1_bar1[1]
            
            # Seconda barra (Indice 1)
            args_2 = calls[1][0]
            pt1_bar2 = args_2[1]
            y_top_2 = pt1_bar2[1]
            
            # Verifica posizione assoluta basata sulle costanti
            assert y_top_1 == START_Y
            expected_y2 = START_Y + (1 * BAR_HEIGHT)
            assert y_top_2 == expected_y2
            
            # Verifica relativa (la seconda deve essere più in basso della prima)
            assert y_top_2 > y_top_1

    def test_numpy_compatibility(self, mock_frame, mock_config_colors):
        """Verifica che la funzione accetti input NumPy senza crashare."""
        probs_np = np.array([0.3, 0.4])
        actions_np = np.array(["Np1", "Np2"])
        
        try:
            render_probability_bars(probs_np, actions_np, mock_frame)
        except Exception as e:
            pytest.fail(f"La funzione ha crashato con input NumPy: {e}")

    def test_color_cycling(self, mock_frame):
        """Verifica che i colori vengano riciclati se ci sono più azioni che colori."""
        # Forniamo solo 2 colori
        custom_colors = [(255, 0, 0), (0, 0, 255)] 
        
        # Chiediamo di disegnare 3 barre
        probs = [0.1, 0.2, 0.3]
        actions = ["1", "2", "3"]
        
        with patch("Dynamic.visualization.cv2") as mock_cv2:
            render_probability_bars(probs, actions, mock_frame, bar_colors=custom_colors)
            
            calls = mock_cv2.rectangle.call_args_list
            
            # Colore barra 1 (Indice 0) -> Colore 0
            color_1 = calls[0][0][3]
            assert color_1 == custom_colors[0]
            
            # Colore barra 2 (Indice 1) -> Colore 1
            color_2 = calls[1][0][3]
            assert color_2 == custom_colors[1]
            
            # Colore barra 3 (Indice 2) -> Deve tornare al Colore 0 (modulo)
            color_3 = calls[2][0][3]
            assert color_3 == custom_colors[0]

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()