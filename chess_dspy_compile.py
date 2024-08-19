import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

from chess_dspy import ChessEngine, load_example_data

import json
from dotenv import load_dotenv
load_dotenv()
    
# task_model = dspy.OpenAI(model="gpt-4o-mini", max_tokens=4000)
# task_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)

task_model = dspy.OpenAI(model="ft:gpt-4o-mini-2024-07-18:devpy:chess-distill:9xrjyG9J", max_tokens=4000)

# task_model = dspy.OpenAI(
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
#     api_key="1aa1f92f2ad33b31d47b752f50737a31e8a312f4546287f8139e691729084702",
#     api_base="https://api.together.xyz/v1/",
#     # stop=["<|eot_id|>","<|eom_id|>"],
#     max_tokens=4000
# )

prompt_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)

dspy.settings.configure(lm=task_model)
program_with_assertions = ChessEngine().activate_assertions()

train_data = load_example_data("chess_finetuning_train.jsonl")
val_data = load_example_data("chess_finetuning_val.jsonl")
# train_data = load_example_data("chess_finetuning_train.jsonl.bak")
# val_data = load_example_data("chess_finetuning_val.jsonl.bak")
train = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in train_data]
val = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in val_data]

# Define hyperparameters:
N = 20 # The number of instructions and fewshot examples that we will generate and optimize over
batches = 50 # The number of optimization trials to be run (we will test out a new combination of instructions and fewshot examples in each trial)
temperature = 1.0 # The temperature configured for generating new instructions


# Set up metrics
NUM_THREADS = 32

# Eval
metric = dspy.evaluate.answer_exact_match
kwargs = dict(num_threads=NUM_THREADS, display_progress=True)
evaluate = Evaluate(devset=val, metric=metric, **kwargs)

# gpt-4o-mini 17.4
# gpt-4o-mini few shot 35.8
# gpt-4o-mini few shot MoA (5x) 25.69
# gpt-4o 28.44
# gpt-4o few shot 62.39

# full
# gpt-4o 23.85
# gpt-4o few shot 65.14

# difficulty filterd [1500+]
# gpt-4o-mini 23.21
# gpt-4o-mini few shot 29.46
# gpt-4o-mini-ft-fixed 28.6
# gpt-4o-mini-ft-first few shot 56.25
# gpt-4o-mini-ft-fixed few shot 59.82
# gpt-4o few shot 63.39


# baseline
# baseline_train_score = evaluate(program,devset=train)
# print(f"Baseline train: {baseline_train_score}")
baseline_val_score = evaluate(program_with_assertions, devset=val)
print(f"Baseline val: {baseline_val_score}")


# Compile
eval_kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
teleprompter = MIPROv2(prompt_model=prompt_model, task_model=task_model, metric=metric, num_candidates=N, init_temperature=temperature, verbose=True)
compiled_program = teleprompter.compile(program_with_assertions, trainset=train, valset=val, num_batches=batches, max_bootstrapped_demos=3,max_labeled_demos=5, eval_kwargs=eval_kwargs)
# compiled_program = teleprompter.compile(compiled_program_base, trainset=train, valset=val, num_batches=batches, max_bootstrapped_demos=3,max_labeled_demos=5, eval_kwargs=eval_kwargs)
compiled_program.save("compiled_chess_cot_ft_student.dspy")

# compiled_program = ChessEngine().activate_assertions()
# compiled_program.load("compiled_chess_cot_ft_student.dspy")

# trained
# fs_train_score = evaluate(compiled_program, devset=train)
# print(f"Few shot compiled train: {fs_train_score}")
fs_val_score, fs_outputs = evaluate(compiled_program, devset=val, return_outputs=True)
print(f"Few shot compiled val: {fs_val_score}")