# -*- coding: utf-8 -*-
"""
Connect4 AI Server - Anvil Uplink (AWS Deployment)

Based on Connect4 - 2.ipynb
Loads trained CNN and Transformer models and serves predictions via Anvil Uplink.

Prerequisites (install on AWS instance before running):
    pip install tensorflow anvil-uplink numpy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import anvil.server


# =============================================================================
# 1. CUSTOM LAYERS (needed for loading the Transformer model)
# =============================================================================

class PositionalIndex(tf.keras.layers.Layer):
    """Generate positional indices for each patch in the sequence."""
    def call(self, x):
        bs = tf.shape(x)[0]  # batch size
        num_patches = tf.shape(x)[1]  # number of patches/tokens
        indices = tf.range(num_patches)
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])


class ClassTokenIndex(tf.keras.layers.Layer):
    """Generate index for the class token (always 0)."""
    def call(self, x):
        bs = tf.shape(x)[0]
        indices = tf.range(1)  # just index 0
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def find_legal(board_2d):
    """
    Find legal moves from a (6,7,2) board representation.
    A column is legal if the top row (row 0) has no piece in either channel.

    Args:
        board_2d: numpy array of shape (6, 7, 2)

    Returns:
        List of legal column indices (0-6)
    """
    legal = [i for i in range(7) if (board_2d[0, i, 0] + board_2d[0, i, 1]) < 0.1]
    return legal


def extract_patches_overlap(boards, patch_rows=4, patch_cols=4, stride_rows=1, stride_cols=1):
    """
    Extract overlapping patches from boards.
    Overlapping patches can better capture connect-4 patterns (4 in a row).

    Args:
        boards: (N, 6, 7, 2) array of board states
        patch_rows: Height of each patch (4 is good for connect-4 patterns)
        patch_cols: Width of each patch
        stride_rows: Vertical stride between patches
        stride_cols: Horizontal stride between patches

    Returns:
        (N, num_patches, patch_dim) array, num_patches, patch_dim
    """
    N, H, W, C = boards.shape  # (N, 6, 7, 2)

    # Calculate number of patches
    n_rows = (H - patch_rows) // stride_rows + 1
    n_cols = (W - patch_cols) // stride_cols + 1
    num_patches = n_rows * n_cols
    patch_dim = patch_rows * patch_cols * C

    patches = np.zeros((N, num_patches, patch_dim), dtype=np.float32)

    idx = 0
    for r in range(n_rows):
        for c in range(n_cols):
            r_start = r * stride_rows
            r_end = r_start + patch_rows
            c_start = c * stride_cols
            c_end = c_start + patch_cols
            patch = boards[:, r_start:r_end, c_start:c_end, :]
            patches[:, idx, :] = patch.reshape(N, -1)
            idx += 1

    return patches, num_patches, patch_dim


# =============================================================================
# 3. MODEL LOADING
# =============================================================================

# Determine model file paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_cnn_final.keras")
TRANSFORMER_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_transformer_final.keras")

# Fallback: check current directory if Results/ path doesn't exist
if not os.path.exists(CNN_MODEL_PATH):
    CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_cnn_final.keras")
if not os.path.exists(TRANSFORMER_MODEL_PATH):
    TRANSFORMER_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_transformer_final.keras")

print("Loading models...")

cnn_model = None
trans_model = None

# Load CNN model
try:
    cnn_model = load_model(CNN_MODEL_PATH, compile=False)
    print(f"CNN model loaded successfully from: {CNN_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load CNN model from {CNN_MODEL_PATH}: {e}")

# Load Transformer model (requires custom layers)
try:
    trans_model = load_model(
        TRANSFORMER_MODEL_PATH,
        custom_objects={
            'PositionalIndex': PositionalIndex,
            'ClassTokenIndex': ClassTokenIndex
        },
        safe_mode=False
    )
    print(f"Transformer model loaded successfully from: {TRANSFORMER_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load Transformer model from {TRANSFORMER_MODEL_PATH}: {e}")

if cnn_model is None and trans_model is None:
    print("ERROR: No models were loaded! Server will not be able to make predictions.")
else:
    print("Model loading complete!")


# =============================================================================
# 4. ANVIL UPLINK CALLABLE
# =============================================================================

@anvil.server.callable
def predict_move(board_list, model_choice):
    """
    Receives board from Anvil (as a flat list) and returns the best column (0-6).

    Args:
        board_list: Flat list representing a (6,7,2) board state (84 values)
        model_choice: "CNN" or "Transformer"

    Returns:
        Best column (0-6) to drop a checker
    """
    # Reshape the flat list into (1, 6, 7, 2) board tensor
    board_array = np.array(board_list, dtype=np.float32).reshape(1, 6, 7, 2)

    if model_choice.upper() == "CNN":
        if cnn_model is None:
            raise Exception("CNN model is not loaded!")
        # Predict and extract from batch dimension: (1, 7) -> (7,)
        predictions = cnn_model.predict(board_array, verbose=0)[0]
    else:
        if trans_model is None:
            raise Exception("Transformer model is not loaded!")
        # Transformer needs patches, not the raw board
        patches, _, _ = extract_patches_overlap(board_array, patch_rows=4, patch_cols=4)
        # Predict and extract from batch dimension: (1, 7) -> (7,)
        predictions = trans_model.predict(patches, verbose=0)[0]

    # Find legal moves using the (6, 7, 2) board representation
    board_2d = board_array[0]  # Shape: (6, 7, 2)
    legal = find_legal(board_2d)

    # Mask illegal moves with -inf
    masked_preds = np.full(7, -np.inf)
    for col in legal:
        masked_preds[col] = predictions[col]

    best_move = int(np.argmax(masked_preds))
    return best_move


# =============================================================================
# 5. START ANVIL UPLINK SERVER
# =============================================================================

# Replace with your actual Anvil Uplink key from Anvil Settings -> Uplink
anvil.server.connect("server_P7WB3AZ3ISIH376TRAQADZB3-FR7JLUM4LM4ONVBQ")
print("Connect 4 AI Server is Online!")
anvil.server.wait_forever()
