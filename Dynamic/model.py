import numpy as np
import tensorflow as tf
from typing import Any, Dict, List, Union, Tuple, Final

from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Conv2D, Dense, 
    Dropout, GlobalAveragePooling2D, Input, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model as keras_load_model

# --- CONFIG IMPORTS ---
from config.dynamic import (
    NUM_NODES, SEQUENCE_LENGTH, INPUT_SHAPE, 
    LEARNING_RATE, LOSS, METRICS, MODEL_PATH
)
from config.gestures import DYNAMIC_ACTIONS
# --- TOPOLOGY CONSTANTS ---
# Indices in the concatenated vector [Pose(0-32), LH(33-53), RH(54-74)]
LH_OFFSET: Final[int] = 33
RH_OFFSET: Final[int] = 54

POSE_CONNECTIONS: Final[List[Tuple[int, int]]] = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), 
    (9,10), (11,12), (11,13), (13,15), (12,14), (14,16),    
    (11,23), (12,24), (23,24), (23,25), (24,26), (25,27), (26,28) 
]

HAND_CONNECTIONS: Final[List[Tuple[int, int]]] = [
    (0,1), (1,2), (2,3), (3,4),         
    (0,5), (5,6), (6,7), (7,8),         
    (0,9), (9,10), (10,11), (11,12),    
    (0,13), (13,14), (14,15), (15,16),  
    (0,17), (17,18), (18,19), (19,20)   
]


def _build_adjacency_matrix() -> tf.Tensor:
    """Constructs the normalized 75x75 adjacency matrix for the skeletal graph."""
    num_nodes = NUM_NODES
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # 1. Intra-body connections
    for i, j in POSE_CONNECTIONS:
        adj[i, j] = adj[j, i] = 1.0
        
    for i, j in HAND_CONNECTIONS:
        # Left Hand
        adj[i + LH_OFFSET, j + LH_OFFSET] = adj[j + LH_OFFSET, i + LH_OFFSET] = 1.0
        # Right Hand
        adj[i + RH_OFFSET, j + RH_OFFSET] = adj[j + RH_OFFSET, i + RH_OFFSET] = 1.0
        
    # 2. Inter-body connections (Wrist Stitching)
    # Pose Left Wrist (15) <-> LH Root (0 + LH_OFFSET)
    adj[15, LH_OFFSET] = adj[LH_OFFSET, 15] = 1.0
    # Pose Right Wrist (16) <-> RH Root (0 + RH_OFFSET)
    adj[16, RH_OFFSET] = adj[RH_OFFSET, 16] = 1.0
    
    # 3. Normalization (D^-1 * A)
    # Add self-loops
    np.fill_diagonal(adj, 1.0)
    
    degree = np.sum(adj, axis=1)
    degree_inv = np.power(degree, -1)
    degree_inv[np.isinf(degree_inv)] = 0.0
    
    adj_norm = np.dot(np.diag(degree_inv), adj)
    
    return tf.constant(adj_norm, dtype=tf.float32)


class DataProcessor(Layer):
    """
    Preprocessing Layer.
    Reshapes flat raw MediaPipe vectors (1662 features) into structured Graph Tensors (75 nodes, 3 coords).
    """
    
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Input shape: (Batch, Seq, 1662)
        
        # 1. Pose: indices 0-132 (33 * 4 [x,y,z,v]) -> extract x,y,z
        pose_raw = inputs[:, :, :132]
        pose_reshaped = tf.reshape(pose_raw, (-1, SEQUENCE_LENGTH, 33, 4))
        pose_xyz = pose_reshaped[:, :, :, :3] 
        
        # 2. Left Hand: indices 1536-1599 (21 * 3)
        lh_raw = inputs[:, :, 1536:1599]
        lh_xyz = tf.reshape(lh_raw, (-1, SEQUENCE_LENGTH, 21, 3))
        
        # 3. Right Hand: indices 1599-end (21 * 3)
        rh_raw = inputs[:, :, 1599:]
        rh_xyz = tf.reshape(rh_raw, (-1, SEQUENCE_LENGTH, 21, 3))
        
        # Concatenate nodes: 33 + 21 + 21 = 75 nodes
        return tf.concat([pose_xyz, lh_xyz, rh_xyz], axis=2)
    
    def get_config(self) -> Dict[str, Any]:
        return super().get_config()


class GraphConv(Layer):
    """
    Spatial Graph Convolution Layer.
    Implements formula: Y = (A * X) * W
    """

    def __init__(
        self, 
        units: int, 
        adjacency_matrix: Union[np.ndarray, tf.Tensor, List], 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        
        # Handle serialization formats (List -> Tensor conversion)
        if isinstance(adjacency_matrix, list):
            self.adj = tf.constant(np.array(adjacency_matrix), dtype=tf.float32)
        elif isinstance(adjacency_matrix, np.ndarray):
             self.adj = tf.constant(adjacency_matrix, dtype=tf.float32)
        else:
            self.adj = adjacency_matrix

    def build(self, input_shape: Tuple[int, ...]) -> None:
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # X * W
        x = tf.matmul(inputs, self.kernel) 
        # A * (XW) -> Einstein summation for batch matrix multiplication
        # v, w: graph nodes (75x75)
        # b: batch, t: time, c: channels
        return tf.einsum('vw,btwc->btvc', self.adj, x)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        
        # Convert Tensor/Numpy to standard list for JSON serialization
        if hasattr(self.adj, 'numpy'):
            adj_list = self.adj.numpy().tolist()
        else:
            adj_list = np.array(self.adj).tolist()
            
        config.update({
            "units": self.units,
            "adjacency_matrix": adj_list
        })
        return config


def _st_gcn_block(
    x: tf.Tensor, 
    adjacency_matrix: tf.Tensor, 
    out_channels: int, 
    stride: int = 1, 
    dropout_rate: float = 0.0
) -> tf.Tensor:
    """Constructs a basic Spatial-Temporal GCN Residual Block."""
    
    # 1. Spatial Graph Convolution
    gcn = GraphConv(out_channels, adjacency_matrix)(x)
    gcn = BatchNormalization()(gcn)
    gcn = Activation('relu')(gcn)
    
    if dropout_rate > 0:
        gcn = Dropout(dropout_rate)(gcn)
    
    # 2. Temporal Convolution (TCN)
    # Kernel (9,1) convolves over time dimension
    tcn = Conv2D(
        out_channels, 
        kernel_size=(9, 1), 
        padding='same', 
        strides=(stride, 1)
    )(gcn)
    tcn = BatchNormalization()(tcn)
    
    # 3. Residual Connection
    # If dimensions change, project 'x' to match 'tcn'
    if x.shape[-1] != out_channels or stride != 1:
        x = Conv2D(
            out_channels, 
            kernel_size=(1, 1), 
            strides=(stride, 1), 
            padding='same'
        )(x)
        x = BatchNormalization()(x)
    
    return Activation('relu')(Add()([tcn, x]))


def create_model() -> Model:
    """Assembles and compiles the ST-GCN Keras Model."""
    inputs = Input(shape=INPUT_SHAPE)
    
    # Preprocessing
    x = DataProcessor()(inputs)
    adj_matrix = _build_adjacency_matrix()
    
    # Backbone
    x = _st_gcn_block(x, adj_matrix, 64)
    x = _st_gcn_block(x, adj_matrix, 128, dropout_rate=0.2)
    x = _st_gcn_block(x, adj_matrix, 128, dropout_rate=0.2)
    
    # Head
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(DYNAMIC_ACTIONS.shape[0], activation='softmax')(x)
    
    model = Model(inputs, outputs, name="ST_GCN_MediaPipe")
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=LOSS,
        metrics=METRICS
    )
    
    return model


def load_model(model_path: str = MODEL_PATH) -> Model:
    """
    Loads the model safely, handling custom layers.
    Falls back to weight-loading if architecture mismatch occurs.
    """
    custom_objects = {
        'DataProcessor': DataProcessor,
        'GraphConv': GraphConv
    }

    try:
        return keras_load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Warning: Full model load failed ({e}). Rebuilding and loading weights.")
        model = create_model()
        model.load_weights(model_path)
        return model