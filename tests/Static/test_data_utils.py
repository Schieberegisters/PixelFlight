import sys
import pytest
import math
import csv
import io  # Added import
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

from Static import data_utils
from Static.data_utils import (
    calculate_distance,
    normalize_landmarks,
    scale_sequence_data,
    save_data_to_csv,
    load_sequences_from_csv,
    flip_frame
)

class TestDataUtils:

    # --- GEOMETRY TESTS ---

    def test_calculate_distance_logic(self):
        """Verifica il calcolo della distanza euclidea 2D."""
        dist = calculate_distance(0, 0, 3, 4)
        assert dist == 5.0
        
        dist_zero = calculate_distance(1.5, 2.5, 1.5, 2.5)
        assert dist_zero == 0.0

    def test_normalize_landmarks_features(self):
        """Verifica la generazione del vettore di feature a 15 dimensioni."""
        points = [MagicMock(x=0.0, y=0.0) for _ in range(21)]
        points[4].x = 1.0
        points[4].y = 0.0
        
        features = normalize_landmarks(points)
        
        assert len(features) == 15
        assert features[0] == 1.0

    def test_flip_frame_call(self):
        """Verifica che cv2.flip venga chiamato con il codice corretto."""
        mock_frame = MagicMock()
        with patch("cv2.flip") as mock_cv_flip:
            flip_frame(mock_frame)
            mock_cv_flip.assert_called_once_with(mock_frame, 1)

    # --- DATA AUGMENTATION TESTS ---

    def test_scale_sequence_data_logic(self):
        """Verifica la scalatura dei dati numerici e la gestione degli ID sequenza."""
        seq_data = [
            {"label": "A", "seq_id": "100", "feat_1": "10.0", "feat_2": "20.0"}
        ]
        factors = [0.5, 2.0]
        start_id = 200
        
        scaled_seqs, next_id = scale_sequence_data(seq_data, factors, start_id)
        
        assert len(scaled_seqs) == 2
        
        seq_0_frame = scaled_seqs[0][0]
        assert seq_0_frame["seq_id"] == "200"
        assert float(seq_0_frame["feat_1"]) == 5.0
        assert float(seq_0_frame["feat_2"]) == 10.0
        
        seq_1_frame = scaled_seqs[1][0]
        assert seq_1_frame["seq_id"] == "201"
        assert float(seq_1_frame["feat_1"]) == 20.0
        
        assert next_id == 202

    def test_scale_sequence_ignores_immutable(self):
        """Verifica che label e seq_id non vengano moltiplicati."""
        seq_data = [{"label": "TEST", "seq_id": "999", "val": "10"}]
        scaled, _ = scale_sequence_data(seq_data, [2.0], 1000)
        
        res = scaled[0][0]
        assert res["label"] == "TEST"
        assert res["seq_id"] == "1000"

    # --- CSV IO TESTS ---

    def test_save_data_to_csv_dict_writer(self):
        """Verifica la scrittura di dizionari su CSV."""
        data = [{"col1": "A", "col2": "B"}]
        path = "dummy.csv"
        
        with patch("builtins.open", mock_open()) as mock_file, \
             patch("os.path.exists", return_value=False):
             
             save_data_to_csv(data, path, mode="w", write_header=True)
             
             handle = mock_file()
             handle.write.assert_any_call("col1,col2\r\n")
             handle.write.assert_any_call("A,B\r\n")

    def test_load_sequences_from_csv_header_detection(self):
        """Verifica il caricamento e il raggruppamento per seq_id."""
        csv_content = (
            "label,seq_id,feat_1\n"
            "GestA,10,0.5\n"
            "GestA,10,0.6\n"
            "GestB,11,0.9\n"
        )
        
        # FIX: Use io.StringIO instead of mock_open because the code uses seek(0)
        # and readline() mixed with iteration, which mock_open doesn't handle natively.
        with patch("builtins.open", return_value=io.StringIO(csv_content)):
            sequences, fields = load_sequences_from_csv("dummy.csv")
            
            assert "label" in fields
            assert len(sequences) == 2 
            
            assert len(sequences[10]) == 2
            assert sequences[10][0]["feat_1"] == "0.5"
            
            assert len(sequences[11]) == 1
            assert sequences[11][0]["label"] == "GestB"

    def test_expand_training_data_flow(self):
        """Verifica il flusso completo di espansione dataset."""
        mock_seqs = {1: [{"label": "A", "seq_id": "1", "val": "10"}]}
        mock_fields = ["label", "seq_id", "val"]
        
        with patch("Static.data_utils.load_sequences_from_csv", return_value=(mock_seqs, mock_fields)) as mock_load, \
             patch("Static.data_utils.scale_sequence_data") as mock_scale, \
             patch("builtins.open", mock_open()) as mock_file:
            
            mock_scale.return_value = ([ [{"label": "A", "seq_id": "2", "val": "20"}] ], 3)
            
            out_path = "out.csv"
            factors = [2.0]
            
            res = data_utils.expand_training_data("in.csv", out_path, factors)
            
            assert res == out_path
            mock_load.assert_called_once()
            mock_scale.assert_called_once()
            mock_file.assert_called_with(out_path, 'w', newline='', encoding='utf-8')

def run_tests_directly() -> None:
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()