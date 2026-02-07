import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tensorflow as tf

from Dynamic.model import (
    create_model,
    load_model,
    _build_adjacency_matrix,
    DataProcessor,
    GraphConv,
    NUM_NODES
)

class TestSTGCNModel:
    """Test suite for Spatio-Temporal Graph Convolutional Network components."""

    def test_adjacency_matrix_properties(self) -> None:
        """Validates mathematical properties of the generated adjacency matrix."""
        adj = _build_adjacency_matrix()
        adj_np = adj.numpy()
        
        # Verify dimensions
        assert adj_np.shape == (NUM_NODES, NUM_NODES)
        
        # Verify self-loops (diagonal elements must be non-zero)
        diag = np.diag(adj_np)
        assert np.all(diag > 0)
        
        # Verify structural symmetry for basic connectivity
        assert adj_np[0, 1] > 0
        assert adj_np[1, 0] > 0

    def test_data_processor_layer_logic(self) -> None:
        """Verifies reshaping and concatenation logic in the preprocessing layer."""
        # Simulated input: Batch=1, Seq=10, Feats=1662
        input_tensor = tf.zeros((1, 10, 1662))
        
        with patch("Dynamic.model.SEQUENCE_LENGTH", 10):
            layer = DataProcessor()
            output = layer(input_tensor)
            
            # Expected: (Batch, Seq, Nodes, Channels) -> (1, 10, 75, 3)
            # 3 Channels represent x, y, z coordinates
            assert output.shape == (1, 10, 75, 3)

    def test_graph_conv_layer_execution(self) -> None:
        """Verifies forward pass execution of the Graph Convolutional layer."""
        # Input: (Batch, Seq, Nodes, Channels) -> (1, 5, 75, 3)
        input_tensor = tf.random.normal((1, 5, 75, 3))
        adj = tf.eye(75)
        
        layer = GraphConv(units=16, adjacency_matrix=adj)
        output = layer(input_tensor)
        
        # Expected: (Batch, Seq, Nodes, Units) -> (1, 5, 75, 16)
        assert output.shape == (1, 5, 75, 16)

    def test_graph_conv_serialization(self) -> None:
        """Ensures the custom GraphConv layer supports Keras serialization."""
        adj = np.eye(75)
        layer = GraphConv(units=32, adjacency_matrix=adj)
        
        config = layer.get_config()
        
        assert config["units"] == 32
        assert "adjacency_matrix" in config
        assert isinstance(config["adjacency_matrix"], list)

    def test_create_model_structure(self) -> None:
        """Validates the assembly and dimensions of the full ST-GCN Keras model."""
        model = create_model()
        
        assert isinstance(model, tf.keras.Model)
        
        # Extract input shape (handling potential list returns in TF)
        input_shape = model.input_shape
        if isinstance(input_shape, list): 
            input_shape = input_shape[0]
            
        assert input_shape[1:] == (20, 1662)
        
        # Validate output dimension against dynamic action classes
        from config.gestures import DYNAMIC_ACTIONS
        output_shape = model.output_shape
        if isinstance(output_shape, list):
            output_shape = output_shape[0]
            
        assert output_shape[1] == len(DYNAMIC_ACTIONS)
        
        # Ensure at least one GraphConv layer is present in the architecture
        assert any(isinstance(layer, GraphConv) for layer in model.layers)

    @patch("Dynamic.model.keras_load_model")
    def test_load_model_custom_objects(self, mock_keras_load: MagicMock) -> None:
        """Verifies that custom layers are correctly registered during model loading."""
        mock_model = MagicMock()
        mock_keras_load.return_value = mock_model
        
        loaded = load_model("fake.h5")
        
        _, kwargs = mock_keras_load.call_args
        custom_objs = kwargs.get("custom_objects", {})
        
        assert "DataProcessor" in custom_objs
        assert "GraphConv" in custom_objs
        assert loaded == mock_model

    @patch("Dynamic.model.keras_load_model", side_effect=ValueError("Architecture mismatch"))
    @patch("Dynamic.model.create_model")
    def test_load_model_fallback(self, mock_create: MagicMock, mock_keras_load: MagicMock) -> None:
        """Verifies fallback to weight-only loading if architecture instantiation fails."""
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        
        loaded = load_model("fake.h5")
        
        mock_create.assert_called_once()
        mock_model.load_weights.assert_called_once_with("fake.h5")
        assert loaded == mock_model

def run_tests_directly() -> None:
    """Execution entry point for direct script testing."""
    project_root = str(Path(__file__).parents[2])
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"--- Running tests for {Path(__file__).name} ---")
    exit_code = pytest.main(["-v", "-p", "no:cacheprovider", __file__])
    sys.exit(exit_code)

if __name__ == "__main__":
    run_tests_directly()