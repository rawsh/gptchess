import dspy
from dspy.evaluate import Evaluate

from chess_dspy import ChessEngine, load_example_data

import json
from dotenv import load_dotenv
load_dotenv()
    
# use strong model
task_model = dspy.OpenAI(model="gpt-4o", max_tokens=4000)
dspy.settings.configure(lm=task_model)

# load data
train_data = load_example_data("chess_finetuning_train.jsonl")
val_data = load_example_data("chess_finetuning_val.jsonl")
train = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in train_data]
val = [dspy.Example(pgn=ex["prompt"].strip(), answer=ex["completion"].strip()).with_inputs("pgn") for ex in val_data]

# Set up metrics
NUM_THREADS = 64

# Eval
metric = dspy.evaluate.answer_exact_match
kwargs = dict(num_threads=NUM_THREADS, display_progress=True)
evaluate = Evaluate(devset=train, metric=metric, **kwargs)


# magic
compiled_program = ChessEngine().activate_assertions()
compiled_program.load("compiled_chess_cot.dspy")

fs_val_score, fs_outputs = evaluate(compiled_program, devset=train, return_outputs=True)
print(f"Few shot compiled val: {fs_val_score}")

def create_jsonl_entry(user, assistant):
    return {
        "messages": [
            {
                "role": "user",
                "content": user
            },
            {
                "role": "assistant",
                "content": assistant
            }
        ]
    }


# prompt = """Given a series of chess moves in Portable Game Notation (PGN) format, your task is to determine and return the correct next move in Standard Algebraic Notation (SAN) format.

# ---

# Follow the following format.

# Pgn: The chess position
# Reasoning: Let's think step by step in order to ${{produce the answer}}. We ...
# Answer: The correct next move in SAN format

# ---

# Pgn: 1. e4 e5 2. Ne2 Bc5 3. Ng3 d6 4. Bc4 h6 5. O-O Nf6 6. d3 O-O 7. Nh5 Bg4 8. Nxf6+ Qxf6 9.
# Reasoning: Let's think step by step in order to Pgn: 1. e4 e5 2. Ne2 Bc5 3. Ng3 d6 4. Bc4 h6 5. O-O Nf6 6. d3 O-O 7. Nh5 Bg4 8. Nxf6+ Qxf6 9. Reasoning: Let's think step by step in order to determine the best move for White. After 8...Qxf6, Black has captured the knight on f6, and now White needs to respond. The most logical move for White is to play 9. Qxg4, capturing the bishop on g4. This move not only gains material but also puts pressure on Black's position.

# ---

# Pgn: 1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8 4. Nf3 Nf6 5. h3 Bf5 6. Bc4 e6 7. d3 Be7 8. Qe2 O-O 9. Bd2 a6 10. g4 Bg6 11. Ne5 b5 12. Bb3 c5 13. h4 h6 14. Nxg6 fxg6 15. g5 Nd5 16. Nxd5 exd5 17. Qe6+ Kh7 18. Bxd5 Bxg5 19. O-O-O Bxd2+ 20. Rxd2 Rf6 21. Qe4 Ra7 22. h5 Re7 23. hxg6+ Kh8 24. Qg2 Qd6 25. Re2 Rxe2 26. Rxh6+ gxh6 27. g7+ Kh7 28.
# Reasoning: Let's think step by step in order to determine the best move for White. The position after 28. is critical as White has a strong initiative and is threatening to deliver checkmate or win material. The last move played by Black was 27...Kh7, which puts the Black king in a precarious position. White has several options to consider, but the most effective move is to play 29. g8=Q+. This move promotes the pawn on g7 to a queen, delivering check to the Black king. The newly promoted queen will also create a significant threat, as it can potentially lead to checkmate on the next move if Black does not respond adequately.

# ---

# Pgn: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4 8. O-O Bxc3 9. d5 Bf6 10. Re1 Ne7 11. Rxe4 d6 12. Bg5 Bxg5 13. Nxg5 h6 14. Nf3 O-O 15. Qe2 Ng6 16. Re1 Bf5 17. Rd4 a6 18. Bd3 Bxd3 19. Qxd3 Qd7 20. h4 Rae8 21. Rxe8 Rxe8 22. h5 Ne5 23. Nxe5 Rxe5 24. g4 Qe7 25. Kg2 Re1 26. Qf5 g6 27. hxg6 fxg6 28. Qxg6+ Qg7 29. Qxg7+ Kxg7 30. Rd2 Kf6 31. f4 Re4 32. Kf3 Rc4 33. b3 Rc5 34. Ke4 Ra5 35. a4 Rc5 36. Rd3 Rc1 37. Rh3 Kg6 38. f5+ Kg5 39. Kf3 Rc3+ 40. Kg2
# Answer: Rxh3

# ---

# Pgn: 1. e4 e5 2. f4 exf4 3. Bc4 d6 4. Nc3 h6 5. d4 g5 6. h4 Bg7 7. hxg5 hxg5 8. Rxh8 Bxh8 9. Qh5 Qf6 10. Nd5 Qxd4 11. Nxc7+ Kd8 12. Nf3 Qxe4+ 13. Be2 Kxc7 14. Qxh8 Ne7 15. Nxg5 Qxg2 16. Bxf4 Bg4 17. O-O-O Qxe2 18. Bxd6+ Kb6 19. Qd4+ Kc6 20.
# Answer: Qc3+

# ---

# Pgn: 1. e4 e5 2. d3 Nc6 3. Be2 d5 4. exd5 Qxd5 5. Bf3 Qd8 6. Bxc6+ bxc6 7. Nf3 Bd6 8. O-O h6 9. Qe2 Qf6 10. d4 Bg4 11. dxe5 Bxf3 12. exf6+ Bxe2 13. Re1 Nxf6 14. Rxe2+ Be7 15. b3 O-O-O 16. Rxe7
# Answer: Rd1+

# ---

# Pgn: {pgn}
# Reasoning: Let's think step by step in order to """

prompt = """Given a series of chess moves in Portable Game Notation (PGN) format, your task is to determine and return the correct next move in Standard Algebraic Notation (SAN) format.

---

Follow the following format.

Pgn: The chess position
Reasoning: Let's think step by step in order to ${{produce the answer}}. We ...
Answer: The correct next move in SAN format

---

Pgn: 1. e4 e5 2. Ne2 Bc5 3. Ng3 d6 4. Bc4 h6 5. O-O Nf6 6. d3 O-O 7. Nh5 Bg4 8. Nxf6+ Qxf6 9.
Reasoning: Let's think step by step in order to Pgn: 1. e4 e5 2. Ne2 Bc5 3. Ng3 d6 4. Bc4 h6 5. O-O Nf6 6. d3 O-O 7. Nh5 Bg4 8. Nxf6+ Qxf6 9. Reasoning: Let's think step by step in order to determine the best move for White. After 8...Qxf6, Black has captured the knight on f6, and now White needs to respond. The most logical move for White is to play 9. Qxg4, capturing the bishop on g4. This move not only gains material but also puts pressure on Black's position.

---

Pgn: 1. e4 d5 2. exd5 Qxd5 3. Nc3 Qd8 4. Nf3 Nf6 5. h3 Bf5 6. Bc4 e6 7. d3 Be7 8. Qe2 O-O 9. Bd2 a6 10. g4 Bg6 11. Ne5 b5 12. Bb3 c5 13. h4 h6 14. Nxg6 fxg6 15. g5 Nd5 16. Nxd5 exd5 17. Qe6+ Kh7 18. Bxd5 Bxg5 19. O-O-O Bxd2+ 20. Rxd2 Rf6 21. Qe4 Ra7 22. h5 Re7 23. hxg6+ Kh8 24. Qg2 Qd6 25. Re2 Rxe2 26. Rxh6+ gxh6 27. g7+ Kh7 28.
Reasoning: Let's think step by step in order to determine the best move for White. The position after 28. is critical as White has a strong initiative and is threatening to deliver checkmate or win material. The last move played by Black was 27...Kh7, which puts the Black king in a precarious position. White has several options to consider, but the most effective move is to play 29. g8=Q+. This move promotes the pawn on g7 to a queen, delivering check to the Black king. The newly promoted queen will also create a significant threat, as it can potentially lead to checkmate on the next move if Black does not respond adequately.

---

Pgn: 1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 7. Nc3 Nxe4 8. O-O Bxc3 9. d5 Bf6 10. Re1 Ne7 11. Rxe4 d6 12. Bg5 Bxg5 13. Nxg5 h6 14. Nf3 O-O 15. Qe2 Ng6 16. Re1 Bf5 17. Rd4 a6 18. Bd3 Bxd3 19. Qxd3 Qd7 20. h4 Rae8 21. Rxe8 Rxe8 22. h5 Ne5 23. Nxe5 Rxe5 24. g4 Qe7 25. Kg2 Re1 26. Qf5 g6 27. hxg6 fxg6 28. Qxg6+ Qg7 29. Qxg7+ Kxg7 30. Rd2 Kf6 31. f4 Re4 32. Kf3 Rc4 33. b3 Rc5 34. Ke4 Ra5 35. a4 Rc5 36. Rd3 Rc1 37. Rh3 Kg6 38. f5+ Kg5 39. Kf3 Rc3+ 40. Kg2
Answer: Rxh3

---

Pgn: 1. e4 e5 2. f4 exf4 3. Bc4 d6 4. Nc3 h6 5. d4 g5 6. h4 Bg7 7. hxg5 hxg5 8. Rxh8 Bxh8 9. Qh5 Qf6 10. Nd5 Qxd4 11. Nxc7+ Kd8 12. Nf3 Qxe4+ 13. Be2 Kxc7 14. Qxh8 Ne7 15. Nxg5 Qxg2 16. Bxf4 Bg4 17. O-O-O Qxe2 18. Bxd6+ Kb6 19. Qd4+ Kc6 20.
Answer: Qc3+

---

Pgn: 1. e4 e5 2. d3 Nc6 3. Be2 d5 4. exd5 Qxd5 5. Bf3 Qd8 6. Bxc6+ bxc6 7. Nf3 Bd6 8. O-O h6 9. Qe2 Qf6 10. d4 Bg4 11. dxe5 Bxf3 12. exf6+ Bxe2 13. Re1 Nxf6 14. Rxe2+ Be7 15. b3 O-O-O 16. Rxe7
Answer: Rd1+

---

Pgn: {pgn}
Reasoning: Let's think step by step in order to """

# write ft data, distill strong model
correct_outputs = [(idx, output) for idx, output in enumerate(fs_outputs) if output[2]]
print(f"Creating examples for {len(correct_outputs)}/{len(fs_outputs)} correct predictions")
output_file = "./chess_finetuning_correct_examples.jsonl"
with open(output_file, 'w') as train_file:
    for idx, output in correct_outputs:
        # prompt = task_model.history[-idx-1]['prompt']
        example, pred, _ = output
        assistant_text = f"{pred.rationale}\n\nAnswer: {pred.answer}"
        json_entry = create_jsonl_entry(prompt.format(pgn=example["pgn"]), assistant_text)
        # json_entry = create_jsonl_entry("", assistant_text)
        train_file.write(json.dumps(json_entry) + '\n')
        train_file.flush()

# print(task_model.history[-1]['prompt'])
# print("\n\n\n------ HIST ------")
# print(task_model.inspect_history(n=1).split("\n\n\n"))