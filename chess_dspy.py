import dspy

import io
import json
import chess
import chess.pgn


def load_example_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def get_signature(predictor):
    return getattr(predictor, 'extended_signature', predictor.signature)

def validate_pgn_move(pgn_board, san_move):
    # Create a board from the PGN
    board = chess.Board()
    pgn = io.StringIO(pgn_board)
    game = chess.pgn.read_game(pgn)
    
    # Apply all moves from the PGN to get the current board state
    for move in game.mainline_moves():
        board.push(move)
    
    # Parse the new move
    try:
        print(str(san_move))
        chess_move = board.parse_san(str(san_move))
    except chess.InvalidMoveError:
        return False, "Invalid move notation"
    except chess.IllegalMoveError:
        return False, "Illegal move"
    except chess.AmbiguousMoveError:
        return False, "SAN is ambigious"
    
    # Check if the move is legal
    if chess_move in board.legal_moves:
        return True, "Move is valid"
    else:
        return False, "Move is not legal in the current position"

class ChessSolver(dspy.Signature):
    """Given a series of chess moves in Portable Game Notation (PGN) format, your task is to determine and return the correct next move in Standard Algebraic Notation (SAN) format."""
    pgn = dspy.InputField(desc="The chess position")
    answer = dspy.OutputField(desc="The correct next move in SAN format")

class ChessEngine(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_move = dspy.ChainOfThought(ChessSolver)
        # self.generate_move = dspy.Predict(ChessSolver)

    def forward(self,pgn):
        gen_pred = self.generate_move(pgn=pgn)
        gen_move = gen_pred.answer
        gen_move = gen_move.split(" ")[-1]
        valid, reason = validate_pgn_move(pgn, gen_move)
        dspy.Suggest(valid, reason)
        if valid:
            print(f"valid:\n{pgn} *{gen_move}")
        if not valid:
            print(f"invalid:\n{pgn} *{gen_move}*\n{reason}")

        return dspy.Prediction(pgn=pgn, answer=gen_move, rationale=gen_pred.rationale)