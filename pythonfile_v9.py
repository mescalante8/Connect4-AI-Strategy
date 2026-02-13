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

    def get_config(self):
        return super().get_config()


class ClassTokenIndex(tf.keras.layers.Layer):
    """Generate index for the class token (always 0)."""
    def call(self, x):
        bs = tf.shape(x)[0]
        indices = tf.range(1)  # just index 0
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])

    def get_config(self):
        return super().get_config()


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


def find_legal_board(board):
    """
    Find legal moves from a (6,7) board representation (+1/-1/0).
    A column is legal if the top row (row 0) is empty.

    Args:
        board: numpy array of shape (6, 7)

    Returns:
        List of legal column indices (0-6)
    """
    legal = [i for i in range(7) if abs(board[0, i]) < 0.1]
    return legal


def update_board(board_temp, color, column):
    """
    Place a checker on the board in the given column.

    Args:
        board_temp: (6, 7) numpy array with +1/-1/0
        color: 'plus' or 'minus'
        column: Column index (0-6) to drop the checker

    Returns:
        Updated board (copy)
    """
    board = board_temp.copy()
    colsum = (abs(board[0, column]) + abs(board[1, column]) + abs(board[2, column])
              + abs(board[3, column]) + abs(board[4, column]) + abs(board[5, column]))
    row = int(5 - colsum)
    if row > -0.5:
        if color == 'plus':
            board[row, column] = 1
        else:
            board[row, column] = -1
    return board


def board_to_2D(board):
    """
    Convert a (6,7) board with +1/-1/0 to a (6,7,2) binary representation.
    Channel 0: 1 where plus player has a piece
    Channel 1: 1 where minus player has a piece

    Args:
        board: numpy array of shape (6, 7)

    Returns:
        numpy array of shape (6, 7, 2)
    """
    out = np.empty(board.shape + (2,), dtype=np.int8)
    np.equal(board, 1, out=out[..., 0])
    np.equal(board, -1, out=out[..., 1])
    return out


def check_for_win(board, col):
    """
    Check if the last move (in the given column) resulted in a win.
    Optimized to only check around the last placed piece.

    Args:
        board: (6, 7) numpy array
        col: Column where the last checker was dropped

    Returns:
        'v-plus', 'v-minus', 'h-plus', 'h-minus', 'd-plus', 'd-minus', or 'nobody'
    """
    nrow = 6
    ncol = 7
    colsum = (abs(board[0, col]) + abs(board[1, col]) + abs(board[2, col])
              + abs(board[3, col]) + abs(board[4, col]) + abs(board[5, col]))
    row = int(6 - colsum)

    # Vertical
    if row + 3 < 6:
        vert = board[row, col] + board[row+1, col] + board[row+2, col] + board[row+3, col]
        if vert == 4:
            return 'v-plus'
        elif vert == -4:
            return 'v-minus'

    # Horizontal (4 possible alignments)
    if col + 3 < 7:
        hor = board[row, col] + board[row, col+1] + board[row, col+2] + board[row, col+3]
        if hor == 4:
            return 'h-plus'
        elif hor == -4:
            return 'h-minus'
    if col - 1 >= 0 and col + 2 < 7:
        hor = board[row, col-1] + board[row, col] + board[row, col+1] + board[row, col+2]
        if hor == 4:
            return 'h-plus'
        elif hor == -4:
            return 'h-minus'
    if col - 2 >= 0 and col + 1 < 7:
        hor = board[row, col-2] + board[row, col-1] + board[row, col] + board[row, col+1]
        if hor == 4:
            return 'h-plus'
        elif hor == -4:
            return 'h-minus'
    if col - 3 >= 0:
        hor = board[row, col-3] + board[row, col-2] + board[row, col-1] + board[row, col]
        if hor == 4:
            return 'h-plus'
        elif hor == -4:
            return 'h-minus'

    # Diagonal: top-left to bottom-right (4 possible alignments)
    if row < 3 and col < 4:
        DR = board[row, col] + board[row+1, col+1] + board[row+2, col+2] + board[row+3, col+3]
        if DR == 4:
            return 'd-plus'
        elif DR == -4:
            return 'd-minus'
    if row-1 >= 0 and col-1 >= 0 and row+2 < 6 and col+2 < 7:
        DR = board[row-1, col-1] + board[row, col] + board[row+1, col+1] + board[row+2, col+2]
        if DR == 4:
            return 'd-plus'
        elif DR == -4:
            return 'd-minus'
    if row-2 >= 0 and col-2 >= 0 and row+1 < 6 and col+1 < 7:
        DR = board[row-2, col-2] + board[row-1, col-1] + board[row, col] + board[row+1, col+1]
        if DR == 4:
            return 'd-plus'
        elif DR == -4:
            return 'd-minus'
    if row-3 >= 0 and col-3 >= 0:
        DR = board[row-3, col-3] + board[row-2, col-2] + board[row-1, col-1] + board[row, col]
        if DR == 4:
            return 'd-plus'
        elif DR == -4:
            return 'd-minus'

    # Diagonal: top-right to bottom-left (4 possible alignments)
    if row+3 < 6 and col-3 >= 0:
        DL = board[row, col] + board[row+1, col-1] + board[row+2, col-2] + board[row+3, col-3]
        if DL == 4:
            return 'd-plus'
        elif DL == -4:
            return 'd-minus'
    if row-1 >= 0 and col+1 < 7 and row+2 < 6 and col-2 >= 0:
        DL = board[row-1, col+1] + board[row, col] + board[row+1, col-1] + board[row+2, col-2]
        if DL == 4:
            return 'd-plus'
        elif DL == -4:
            return 'd-minus'
    if row-2 >= 0 and col+2 < 7 and row+1 < 6 and col-1 >= 0:
        DL = board[row-2, col+2] + board[row-1, col+1] + board[row, col] + board[row+1, col-1]
        if DL == 4:
            return 'd-plus'
        elif DL == -4:
            return 'd-minus'
    if row-3 >= 0 and col+3 < 7:
        DL = board[row-3, col+3] + board[row-2, col+2] + board[row-1, col+1] + board[row, col]
        if DL == 4:
            return 'd-plus'
        elif DL == -4:
            return 'd-minus'

    return 'nobody'


def display_board(board):
    """
    Display the board as ASCII art. X = plus (+1), O = minus (-1).

    Args:
        board: (6, 7) numpy array
    """
    horizontal_line = '-' * (7 * 5 + 8)
    blank_line = '|' + '     |' * 7
    print('   0     1     2     3     4     5     6')
    print(horizontal_line)
    for row in range(6):
        print(blank_line)
        this_line = '|'
        for col in range(7):
            if board[row, col] == 0:
                this_line += '     |'
            elif board[row, col] == 1:
                this_line += '  X  |'
            else:
                this_line += '  O  |'
        print(this_line)
        print(blank_line)
        print(horizontal_line)
    print('   0     1     2     3     4     5     6')


def model_predict_move(board, color, model_choice):
    """
    Predict the best move using the specified model.
    Handles board conversion, perspective normalization, and illegal move masking.

    Args:
        board: (6, 7) numpy array with +1 (plus), -1 (minus), 0 (empty)
        color: 'plus' or 'minus' - which player is making the move
        model_choice: 'CNN' or 'Transformer'

    Returns:
        Best column (0-6) to drop a checker
    """
    # Convert board to 2D representation (6,7,2)
    board_2d = board_to_2D(board)

    # If minus player, swap perspective so NN always sees plus-to-move
    if color == 'minus':
        board_2d = board_2d[..., ::-1]

    # Add batch dimension and convert to float
    board_batch = board_2d[np.newaxis, ...].astype(np.float32)

    if model_choice.upper() == "CNN":
        if cnn_model is None:
            raise Exception("CNN model is not loaded!")
        predictions = cnn_model.predict(board_batch, verbose=0)[0]
    else:
        if trans_model is None:
            raise Exception("Transformer model is not loaded!")
        patches, _, _ = extract_patches_overlap(board_batch, patch_rows=4, patch_cols=4)
        predictions = trans_model.predict(patches, verbose=0)[0]

    # Find legal moves and mask illegal ones
    legal = find_legal_board(board)
    masked_preds = np.full(7, -np.inf)
    for col in legal:
        masked_preds[col] = predictions[col]

    return int(np.argmax(masked_preds))


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

# Determine model file paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_MODEL_PATH = "connect4_cnn_final.keras"
TRANSFORMER_MODEL_PATH = "connect4_transformer_test.keras"

# # Fallback: check script directory if not found in current directory
# if not os.path.exists(CNN_MODEL_PATH):
#     CNN_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_cnn_final.keras")
# if not os.path.exists(TRANSFORMER_MODEL_PATH):
#     TRANSFORMER_MODEL_PATH = os.path.join(SCRIPT_DIR, "connect4_transformer_test.keras")

print("Loading models...")

cnn_model = None
trans_model = None

# Load CNN model
try:
    cnn_model = load_model(CNN_MODEL_PATH, compile=False)
    print(f"CNN model loaded successfully from: {CNN_MODEL_PATH}")
except TypeError:
    try:
        cnn_model = load_model(CNN_MODEL_PATH)
        print(f"CNN model loaded successfully from: {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"Warning: Could not load CNN model from {CNN_MODEL_PATH}: {e}")
except Exception as e:
    print(f"Warning: Could not load CNN model from {CNN_MODEL_PATH}: {e}")

# Load Transformer model (requires custom layers)
custom_dict = {
    "PositionalIndex": PositionalIndex,
    "ClassTokenIndex": ClassTokenIndex
}

try:
    # trans_model = load_model(
    #     TRANSFORMER_MODEL_PATH,
    #     custom_objects=custom_dict,
    #     compile=False,
    #     safe_mode=False
    # )

    # 1. Force the loading process to happen on CPU
    with tf.device('/CPU:0'):
        print("Loading model to CPU...")
        # Make sure to include your custom objects and safe_mode=False
        transformer_model = load_model(
            '/content/drive/MyDrive/Dataset/connect4_transformer_test.keras',
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
def predict_move(board_list, model_choice, color):
    """
    Receives board from Anvil (as a flat list) and returns the best column (0-6).

    Args:
        board_list: Flat list representing a (6,7,2) board state (84 values)
        model_choice: "CNN" or "Transformer"
        color: "plus" or "minus" - which player the AI is playing as

    Returns:
        Best column (0-6) to drop a checker
    """
    # Reshape the flat list into (1, 6, 7, 2) board tensor
    board_array = np.array(board_list, dtype=np.float32).reshape(1, 6, 7, 2)

    # Find legal moves BEFORE flipping (legal moves are the same regardless of perspective)
    board_2d = board_array[0]  # Shape: (6, 7, 2)
    legal = find_legal(board_2d)

    # Flip perspective if playing as minus.
    # The models were trained with all boards normalized to "plus-to-move" perspective,
    # so channel 0 = current player's pieces, channel 1 = opponent's pieces.
    # If the AI is minus, we swap channels so the model sees itself as "plus".
    if color.lower() == "minus":
        board_array = board_array[..., ::-1]

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

    # Mask illegal moves with -inf
    masked_preds = np.full(7, -np.inf)
    for col in legal:
        masked_preds[col] = predictions[col]

    best_move = int(np.argmax(masked_preds))
    return best_move


# =============================================================================
# 5. PLAY AGAINST MODEL (Local Testing)
# =============================================================================

def play_against_model(model_choice, human_color='plus'):
    """
    Play a game of Connect4 against the AI bot in the terminal.
    Unified function that works with either the CNN or Transformer model.

    Args:
        model_choice: 'CNN' or 'Transformer'
        human_color: 'plus' or 'minus' - which color the human plays
    """
    board = np.zeros((6, 7))
    winner = 'nobody'
    current_player = 'plus'  # plus always goes first

    bot_color = 'minus' if human_color == 'plus' else 'plus'
    bot_name = f"{model_choice.upper()} bot"

    display_board(board)

    while winner == 'nobody':
        legal = find_legal_board(board)
        if len(legal) == 0:
            print("It's a tie!")
            return 'tie'

        if current_player == human_color:
            # Human's turn
            move = input(f'Your turn ({human_color}). Pick a column (0-6): ')
            try:
                move = int(move)
                if move not in legal:
                    print(f"Illegal move! Legal moves are: {legal}")
                    continue
            except ValueError:
                print("Please enter a number 0-6")
                continue
        else:
            # Bot's turn
            move = model_predict_move(board, current_player, model_choice)
            print(f"{bot_name} plays column: {move}")

        board = update_board(board, current_player, move)
        display_board(board)
        winner = check_for_win(board, move)

        # Switch player
        current_player = 'minus' if current_player == 'plus' else 'plus'

    if 'plus' in winner:
        winning_color = 'plus'
    else:
        winning_color = 'minus'

    if winning_color == human_color:
        print("Congratulations! You won!")
    else:
        print(f"{bot_name} wins!")

    return winner


# Uncomment to play locally:
# play_against_model('CNN', human_color='plus')
# play_against_model('Transformer', human_color='plus')


# =============================================================================
# 6. START ANVIL UPLINK SERVER
# =============================================================================

# Replace with your actual Anvil Uplink key from Anvil Settings -> Uplink
anvil.server.connect("server_P7WB3AZ3ISIH376TRAQADZB3-FR7JLUM4LM4ONVBQ")
print("Connect 4 AI Server is Online!")
anvil.server.wait_forever()
