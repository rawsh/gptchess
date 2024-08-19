import os
import dspy
from dspy.evaluate import Evaluate

from chess_dspy import ChessEngine, ChessSolver, get_signature, load_example_data

def get_program_number(filename):
    # Extract the number from the filename
    return int(filename.split('_')[-1].split('.`')[0])

# ensemble = [prog for *_, prog in compiled_program.candidate_programs[:10]]
# for idx, prog in enumerate(ensemble):
#     prog.save(f'checkpoints/chess_fewshot_cot_{idx}.json')

# top_programs = []

# for trial_num, trial in compiled_program.trial_logs.items():
#     if trial["score"] > 0 and not trial["pruned"] and trial["full_eval"]:
#         top_programs.append((trial["score"], trial["program"]))

# top_programs.sort(reverse=True, key=lambda x: x[0])

# for i, (score, program) in enumerate(top_programs[:5], 1):
#     print(f"Program {i} | Score: {score}")
#     program.save(f'checkpoints/chess_fewshot_cot_{i}.json')
#     for j, predictor in enumerate(program.predictors(), 1):
#         print(f"Prompt {j}: {get_signature(predictor).instructions}")
#     print()

# Load all available programs from disk
program_tuples = []
checkpoints_dir = 'checkpoints'
for filename in os.listdir(checkpoints_dir):
    if filename.startswith('chess_fewshot_cot_') and filename.endswith('.json'):
        file_path = os.path.join(checkpoints_dir, filename)
        program = ChessEngine().activate_assertions()
        program.load(file_path)
        program_number = get_program_number(filename)
        program_tuples.append((program_number, program))

# Sort programs by their number (assuming lower is better)
program_tuples.sort(key=lambda x: x[0])

# Print information about each program
for i, (program_number, program) in enumerate(program_tuples, 1):
    print(f"Program {program_number}")
    for j, predictor in enumerate(program.predictors(), 1):
        print(f"Prompt {j}: {get_signature(predictor).instructions}")
    print()

compiled_programs = [program for _, program in program_tuples]
print(f"\nTotal programs loaded: {len(program_tuples)}")
print("Note: Programs are ranked based on their filename numbers, with lower numbers assumed to be better.")


class ChessMoA(dspy.Module):
    def __init__(self, top_compiled_programs):
        super().__init__()
        self.compare_answers = dspy.MultiChainComparison(ChessSolver)
        self.top_programs = top_compiled_programs

    def forward(self,pgn):
        completions = []
        for program in self.top_programs:
            gen_pred = program(pgn=pgn)
            completions.append(gen_pred)

        # dedupe
        completions = list(set(completions))
        final_pred = self.compare_answers(completions, pgn=pgn)
        final_move = final_pred.answer
        final_move = final_move.split(" ")[-1]
        print(f"Final Predicted Move (after comparison): {final_pred.answer}")
        print(f"Final Rationale: {final_pred.rationale}")
        return dspy.Prediction(pgn=pgn, answer=final_move)
    
train_data = load_example_data("chess_finetuning_train.jsonl")
val_data = load_example_data("chess_finetuning_val.jsonl")

train = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in train_data]
val = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in val_data]

# Set up metrics
NUM_THREADS = 32

# Eval
metric = dspy.evaluate.answer_exact_match
kwargs = dict(num_threads=NUM_THREADS, display_progress=True)
evaluate = Evaluate(devset=val, metric=metric, **kwargs)


chess_moa = ChessMoA(top_compiled_programs=compiled_programs)
chess_moa_val_score = evaluate(chess_moa, devset=val)
print(f"Chess MoA val: {chess_moa_val_score}")