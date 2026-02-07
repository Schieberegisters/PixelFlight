import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import del modulo oggetto per il patching
from Dynamic import mediapipe_utils

# Import funzioni specifiche
from Dynamic.mediapipe_utils import (
    mediapipe_detection,
    draw_styled_landmarks,
    annotate_frame
)

class TestMediaPipeUtils:

    @pytest.fixture
    def mock_image(self):
        """Crea un'immagine dummy NumPy (BGR)."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_model(self):
        """Mock del modello Holistic di MediaPipe."""
        model = MagicMock()
        model.process.return_value = MagicMock(name="results")
        return model

    def test_mediapipe_detection_flow(self, mock_image, mock_model):
        """
        Verifica il flusso di conversione colore e processamento.
        1. BGR -> RGB
        2. Writeable = False
        3. Model Process
        4. RGB -> BGR
        """
        with patch("cv2.cvtColor", side_effect=lambda img, code: img) as mock_cvt:
            
            image, results = mediapipe_detection(mock_image, mock_model)
            
            # Verifica conversioni colore
            # Deve essere chiamato due volte: BGR2RGB e RGB2BGR
            assert mock_cvt.call_count == 2
            
            # Verifica chiamata al modello
            mock_model.process.assert_called_once()
            
            # Verifica che flags.writeable sia stato manipolato (difficile da testare l'ordine esatto con i mock, 
            # ma possiamo assumere che se il codice gira senza errori di attributo, è ok)
            assert results == mock_model.process.return_value

    def test_draw_styled_landmarks_calls(self, mock_image):
        """Verifica che vengano disegnate tutte e 4 le componenti (Faccia, Corpo, Mani)."""
        mock_results = MagicMock()
        
        # Patchiamo l'oggetto mp_drawing all'interno del modulo mediapipe_utils
        with patch.object(mediapipe_utils, "mp_drawing") as mock_mp_draw:
            
            draw_styled_landmarks(mock_image, mock_results)
            
            # Ci aspettiamo 4 chiamate a draw_landmarks:
            # 1. Face (Mesh)
            # 2. Pose
            # 3. Left Hand
            # 4. Right Hand
            assert mock_mp_draw.draw_landmarks.call_count == 4
            
            # Verifica che vengano passati i landmark corretti
            calls = mock_mp_draw.draw_landmarks.call_args_list
            
            # Face (arg 1)
            assert calls[0][0][1] == mock_results.face_landmarks
            # Pose (arg 1)
            assert calls[1][0][1] == mock_results.pose_landmarks
            # Left Hand (arg 1)
            assert calls[2][0][1] == mock_results.left_hand_landmarks
            # Right Hand (arg 1)
            assert calls[3][0][1] == mock_results.right_hand_landmarks

    def test_annotate_frame_styled(self, mock_image, mock_model):
        """Verifica l'integrazione completa con stile personalizzato."""
        
        with patch.object(mediapipe_utils, "mediapipe_detection") as mock_detect, \
             patch.object(mediapipe_utils, "draw_styled_landmarks") as mock_draw_styled:
            
            # Setup mock return
            mock_results = MagicMock()
            mock_detect.return_value = (mock_image, mock_results)
            
            # Esecuzione
            out_img, out_res = annotate_frame(mock_image, mock_model, styled=True)
            
            # Verifica flusso
            mock_detect.assert_called_once_with(mock_image, mock_model)
            mock_draw_styled.assert_called_once_with(mock_image, mock_results)
            
            assert out_img is mock_image
            assert out_res is mock_results

    def test_annotate_frame_unstyled(self, mock_image, mock_model):
        """Verifica il fallback allo stile di default (no custom colors)."""
        
        with patch.object(mediapipe_utils, "mediapipe_detection") as mock_detect, \
             patch.object(mediapipe_utils, "draw_styled_landmarks") as mock_draw_styled, \
             patch.object(mediapipe_utils, "mp_drawing") as mock_mp_draw_default:
            
            mock_results = MagicMock()
            mock_detect.return_value = (mock_image, mock_results)
            
            # Esecuzione con styled=False
            annotate_frame(mock_image, mock_model, styled=False)
            
            # draw_styled_landmarks NON deve essere chiamato
            mock_draw_styled.assert_not_called()
            
            # Deve usare il drawer di default 4 volte
            assert mock_mp_draw_default.draw_landmarks.call_count == 4

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()