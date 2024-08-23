import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

from chess_dspy import ChessEngine, load_example_data

import json
from dotenv import load_dotenv
load_dotenv()
    
# task_model = dspy.OpenAI(model="gpt-4o-mini", max_tokens=4000)
# task_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)

# finetune all
# task_model = dspy.OpenAI(model="ft:gpt-4o-mini-2024-07-18:devpy:chess-distill:9xrjyG9J", max_tokens=4000)

# finetune all (new 100 sample)
# task_model = dspy.OpenAI(model="ft:gpt-4o-mini-2024-07-18:devpy:puzzlegod-100ex:9zKtUCDG", max_tokens=4000)

# finetune it1 (129 samples, gpt4o-ft outputs)
# task_model = dspy.OpenAI(model="ft:gpt-4o-mini-2024-07-18:devpy:puzzlegod-it1-129ex:9zLUVZ9j", max_tokens=4000)

# finetune all + 1500
# task_model = dspy.OpenAI(model="ft:gpt-4o-mini-2024-07-18:devpy:chess-distill-cont-1500:9y06KXVV", max_tokens=4000)

# finetune all gpt4o
# task_model = dspy.OpenAI(model="ft:gpt-4o-2024-08-06:devpy:chess-all:9zJ5Y9y8", max_tokens=4000)

# finetune all (new 100 sample) gpt4o
task_model = dspy.OpenAI(model="ft:gpt-4o-2024-08-06:devpy:puzzlegod-100ex:9zKwkawo", max_tokens=4000)

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
NUM_THREADS = 64

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



# full NEW-----
# gpt-4o-mini-ft-1500 few shot 59.91
# gpt-4o-mini-ft-sample few shot 65.64
# gpt-4o-mini-ft-sample FRESH few shot 66.52
# gpt-4o-mini-ft-it1 few shot 
# gpt-4o few shot 63.88
# gpt-4o-ft-1500 few shot 72.25
# gpt-4o-ft-sample few shot 71.37
# gpt-4o-ft-sample FRESH few shot 63.0
# -------------




# difficulty filtered [1500+]
# gpt-4o-mini 23.21
# gpt-4o-mini few shot 29.46
# gpt-4o-mini-ft-fixed 28.6
# gpt-4o-mini-ft-first few shot 56.25
# gpt-4o-mini-ft-fixed few shot 59.82
# gpt-4o-mini-ft-fix fresh-compiled few shot 55.36
# gpt-4o few shot 63.39
# gpt-4o fresh-compiled few shot 55.36

# baseline
# baseline_train_score = evaluate(program,devset=train)
# print(f"Baseline train: {baseline_train_score}")
# baseline_val_score = evaluate(program_with_assertions, devset=val)
# print(f"Baseline val: {baseline_val_score}")

# Compile
# eval_kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=0)
# teleprompter = MIPROv2(prompt_model=prompt_model, task_model=task_model, metric=metric, num_candidates=N, init_temperature=temperature, verbose=True)
# compiled_program = teleprompter.compile(program_with_assertions, trainset=train, valset=val, num_batches=batches, max_bootstrapped_demos=3,max_labeled_demos=5, eval_kwargs=eval_kwargs)
# compiled_program.save("compiled_chess_cot_ft_student.dspy")

compiled_program = ChessEngine().activate_assertions()
compiled_program.load("compiled_chess_cot_ft_student.dspy")
# compiled_program = ChessEngine()
# compiled_program.load("compiled_chess_cot.dspy")

# trained
# import random
# train_sample = random.sample(train, 100)
# fs_train_score = evaluate(compiled_program, devset=train)
# print(f"Few shot compiled train: {fs_train_score}")
fs_val_score, fs_outputs = evaluate(compiled_program, devset=val, return_outputs=True)
print(f"Few shot compiled val: {fs_val_score}")

# compiled_program.generate_move.signature = "test"


# print(compiled_program.named_parameters())
# compiled_program(train[-1].pgn)

# print(compiled_program.dump_state())

# print(task_model.history[-1]['prompt'])
# task_model.inspect_history(n=1)