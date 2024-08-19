import json
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
import io

import chess
import chess.pgn

from dotenv import load_dotenv
load_dotenv()

def load_example_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
# task_model = dspy.OpenAI(model="gpt-4o-mini", max_tokens=4000)
task_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)

# task_model = dspy.OpenAI(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#     api_key="1aa1f92f2ad33b31d47b752f50737a31e8a312f4546287f8139e691729084702",
#     api_base="https://api.together.xyz/v1/",
#     # stop=["<|eot_id|>","<|eom_id|>"],
#     max_tokens=4000
# )

prompt_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)

dspy.settings.configure(lm=task_model)


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
    move = dspy.OutputField(desc="The correct next move in SAN format")

class ChessEngine(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_move = dspy.ChainOfThought(ChessSolver)
        # self.generate_move = dspy.Predict(ChessSolver)

    def forward(self,pgn):
        gen_move = self.generate_move(pgn=f"{pgn} ").move
        gen_move = gen_move.split(" ")[-1]
        valid, reason = validate_pgn_move(pgn, gen_move)
        dspy.Suggest(valid, reason)
        if valid:
            print(f"valid:\n{pgn} *{gen_move}")
        if not valid:
            print(f"invalid:\n{pgn} *{gen_move}*\n{reason}")

        return dspy.Prediction(pgn=pgn, answer=gen_move)

program_with_assertions = ChessEngine().activate_assertions()

train_data = load_example_data("chess_finetuning_train.jsonl")
val_data = load_example_data("chess_finetuning_val.jsonl")

train = [dspy.Example(pgn=ex["prompt"], answer=ex["completion"]).with_inputs("pgn") for ex in train_data]
val = [dspy.Example(pgn=ex["prompt"], answer=ex["completion"]).with_inputs("pgn") for ex in val_data]

# Define hyperparameters:
N = 20 # The number of instructions and fewshot examples that we will generate and optimize over
batches = 50 # The number of optimization trials to be run (we will test out a new combination of instructions and fewshot examples in each trial)
temperature = 1.0 # The temperature configured for generating new instructions

metric = dspy.evaluate.answer_exact_match

# Set up metrics
NUM_THREADS = 32

# Eval
kwargs = dict(num_threads=NUM_THREADS, display_progress=True)
evaluate = Evaluate(devset=val, metric=metric, **kwargs)

# gpt-4o-mini 17.4
# gpt-4o-mini few shot 35.8

# gpt-4o 28.44
# gpt-4o few shot 62.39


# # baseline
# # baseline_train_score = evaluate(program,devset=train)
# # print(f"Baseline train: {baseline_train_score}")
baseline_val_score = evaluate(program_with_assertions, devset=val)
print(f"Baseline val: {baseline_val_score}")

# Compile
# eval_kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
# teleprompter = MIPROv2(prompt_model=prompt_model, task_model=task_model, metric=metric, num_candidates=N, init_temperature=temperature, verbose=True)
# compiled_program = teleprompter.compile(program_with_assertions, trainset=train, valset=val, num_batches=batches, max_bootstrapped_demos=3,max_labeled_demos=5, eval_kwargs=eval_kwargs)
# compiled_program.save("compiled_chess.dspy")


compiled_program = ChessEngine().activate_assertions()
compiled_program.load("compiled_chess_cot.dspy")
# trained
# fs_train_score = evaluate(program,devset=train)
# print(f"Few shot compiled train: {fs_train_score}")
fs_val_score = evaluate(compiled_program, devset=val)
print(f"Few shot compiled val: {fs_val_score}")