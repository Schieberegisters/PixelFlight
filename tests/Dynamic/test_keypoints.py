import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

from Dynamic.keypoints import extract_keypoints, TOTAL_LEN, POSE_LEN, FACE_LEN, HAND_LEN

class TestKeypoints:

    @pytest.fixture
    def mock_results(self):
        """Crea un oggetto results mock con strutture MediaPipe simulate."""
        results = MagicMock()
        
        # Helper per creare una lista di landmark mock
        def create_landmarks(count, has_visibility=False):
            landmarks = []
            for i in range(count):
                lm = MagicMock()
                lm.x = 0.1 * i
                lm.y = 0.2 * i
                lm.z = 0.3 * i
                if has_visibility:
                    lm.visibility = 0.9
                landmarks.append(lm)
            
            mock_obj = MagicMock()
            mock_obj.landmark = landmarks
            return mock_obj

        # Setup dei componenti (inizialmente tutti presenti)
        results.pose_landmarks = create_landmarks(33, has_visibility=True)
        results.face_landmarks = create_landmarks(468)
        results.left_hand_landmarks = create_landmarks(21)
        results.right_hand_landmarks = create_landmarks(21)
        
        return results

    def test_extract_keypoints_full_detection(self, mock_results):
        """Verifica l'estrazione quando tutti i landmark sono rilevati."""
        kp = extract_keypoints(mock_results)
        
        assert isinstance(kp, np.ndarray)
        assert kp.shape == (TOTAL_LEN,)
        # Verifica che i primi valori (Pose) siano corretti
        # Pose Landmark 0: x=0, y=0, z=0, v=0.9
        assert kp[0] == 0.0
        assert kp[3] == 0.9

    def test_extract_keypoints_partial_detection(self, mock_results):
        """Verifica che le parti mancanti vengano riempite con zeri."""
        # Simuliamo che la faccia e la mano sinistra non siano visibili
        mock_results.face_landmarks = None
        mock_results.left_hand_landmarks = None
        
        kp = extract_keypoints(mock_results)
        
        assert kp.shape == (TOTAL_LEN,)
        
        # Segmento Pose (0 a POSE_LEN) deve avere dati
        assert np.any(kp[:POSE_LEN] != 0)
        
        # Segmento Face (POSE_LEN a POSE_LEN + FACE_LEN) deve essere zero
        face_segment = kp[POSE_LEN : POSE_LEN + FACE_LEN]
        assert np.all(face_segment == 0)
        
        # Segmento Mano Destra (fine del vettore) deve avere dati
        assert np.any(kp[-HAND_LEN:] != 0)

    def test_extract_keypoints_all_none(self):
        """Verifica il caso estremo in cui non viene rilevato nulla."""
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        mock_results.face_landmarks = None
        mock_results.left_hand_landmarks = None
        mock_results.right_hand_landmarks = None
        
        kp = extract_keypoints(mock_results)
        
        assert kp.shape == (TOTAL_LEN,)
        assert np.all(kp == 0)

    def test_extract_keypoints_shape_mismatch_error(self, mock_results):
        """Verifica che venga sollevato un errore se il numero di landmark è errato."""
        # Manomettiamo il numero di landmark della faccia (es. 10 invece di 468)
        bad_face = MagicMock()
        bad_face.landmark = [MagicMock()] * 10
        mock_results.face_landmarks = bad_face
        
        with pytest.raises(ValueError, match="Vector shape mismatch"):
            extract_keypoints(mock_results)

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()