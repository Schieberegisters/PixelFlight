import sys
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import MagicMock, patch

# --- SETUP PATHS ---
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2]))

# Import module object
from Dynamic import model as stgcn_module
from Dynamic.model import (
    create_model, 
    load_model, 
    _build_adjacency_matrix, 
    DataProcessor, 
    GraphConv,
    NUM_NODES
)

class TestSTGCNModel:

    # --- TEST COMPONENTI INTERNI ---

    def test_adjacency_matrix_properties(self):
        """Verifica le proprietà matematiche della matrice di adiacenza."""
        adj = _build_adjacency_matrix()
        adj_np = adj.numpy()
        
        # 1. Shape deve essere (75, 75)
        assert adj_np.shape == (NUM_NODES, NUM_NODES)
        
        # 2. Diagonale deve essere non-zero (self-loops)
        diag = np.diag(adj_np)
        assert np.all(diag > 0)
        
        # 3. Simmetria strutturale
        assert adj_np[0, 1] > 0
        assert adj_np[1, 0] > 0

    def test_data_processor_layer_logic(self):
        """Verifica il reshaping e la concatenazione del layer di preprocessing."""
        # Input simulato: Batch=1, Seq=10, Feats=1662
        input_tensor = tf.zeros((1, 10, 1662))
        
        # FIX: Patch SEQUENCE_LENGTH to match the test input (10)
        # We patch it inside the module where it is used
        with patch("Dynamic.model.SEQUENCE_LENGTH", 10):
            layer = DataProcessor()
            output = layer(input_tensor)
            
            # Output atteso: (Batch, Seq, Nodes, Channels) -> (1, 10, 75, 3)
            # Channels = 3 because x,y,z are extracted (visibility is dropped for pose)
            assert output.shape == (1, 10, 75, 3)

    def test_graph_conv_layer_execution(self):
        """Verifica che il layer di convoluzione su grafo esegua il passaggio in avanti."""
        # Input: (Batch, Seq, Nodes, Channels) -> (1, 5, 75, 3)
        input_tensor = tf.random.normal((1, 5, 75, 3))
        
        adj = tf.eye(75)
        
        layer = GraphConv(units=16, adjacency_matrix=adj)
        output = layer(input_tensor)
        
        # Output atteso: (Batch, Seq, Nodes, Units) -> (1, 5, 75, 16)
        assert output.shape == (1, 5, 75, 16)

    def test_graph_conv_serialization(self):
        """Verifica che get_config permetta la serializzazione del layer custom."""
        adj = np.eye(75)
        layer = GraphConv(units=32, adjacency_matrix=adj)
        
        config = layer.get_config()
        
        assert config["units"] == 32
        assert "adjacency_matrix" in config
        assert isinstance(config["adjacency_matrix"], list)

    # --- TEST MODELLO COMPLETO ---

    def test_create_model_structure(self):
        """Verifica che il modello Keras venga assemblato correttamente."""
        # We need to ensure global constants match default config here
        # Assuming config.dynamic has SEQUENCE_LENGTH=20 usually
        
        model = create_model()
        
        assert isinstance(model, tf.keras.Model)
        
        # Check input shape (excluding batch)
        # model.input_shape might return a tuple or list depending on TF version
        input_shape = model.input_shape
        if isinstance(input_shape, list): 
            input_shape = input_shape[0]
            
        # Check standard config dimensions (20, 1662)
        # Note: If your config differs locally, update these values
        assert input_shape[1:] == (20, 1662)
        
        # Check output shape
        from config.gestures import DYNAMIC_ACTIONS
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
            
        assert output_shape[1] == len(DYNAMIC_ACTIONS)
        
        # Verify custom layer presence
        has_graph_conv = any(isinstance(layer, GraphConv) for layer in model.layers)
        assert has_graph_conv

    @patch("Dynamic.model.keras_load_model")
    def test_load_model_custom_objects(self, mock_keras_load):
        """Verifica che load_model passi correttamente i custom_objects."""
        mock_model = MagicMock()
        mock_keras_load.return_value = mock_model
        
        loaded = load_model("fake.h5")
        
        args, kwargs = mock_keras_load.call_args
        assert args[0] == "fake.h5"
        
        custom_objs = kwargs.get("custom_objects", {})
        assert "DataProcessor" in custom_objs
        assert "GraphConv" in custom_objs
        assert loaded == mock_model

    @patch("Dynamic.model.keras_load_model", side_effect=ValueError("Architecture mismatch"))
    @patch("Dynamic.model.create_model")
    def test_load_model_fallback(self, mock_create, mock_keras_load):
        """Verifica il fallback sui pesi se il caricamento dell'architettura fallisce."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        loaded = load_model("fake.h5")
        
        mock_create.assert_called_once()
        mock_model.load_weights.assert_called_once_with("fake.h5")
        assert loaded == mock_model

def run_tests_directly() -> None:
    """Entry point per l'esecuzione diretta dello script."""
    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()