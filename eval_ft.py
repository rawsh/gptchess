# all elo
# babbage: Accuracy: 54.42%
# davinci: Accuracy: 74.19%
# davinci ft 1: Accuracy: 75.35%
# davinci ft 5: Accuracy: 80.82%


# new set all elo
# babbage: 61.23%
# davinci: 74.45%
# davinci-ft: 76.21%
# gpt-4o-ft-1: 58.59%
# gpt-4o-mini-ft: 54.19%
# gpt-4o: 55.07%
# gpt-4o-mini-ft-it1: 54.19%
# gpt-4o-mini: 25.99%


# 2000+ elo
# davinci: 61.90%
# davinci ft 1: 61.90%
# davinci ft 2: 57.14%
# davinci ft 3: 58.73%
# davinci ft 4: 60.32%
# davinci ft 5: 65.08%

import json
import multiprocessing
from openai import OpenAI

def load_validation_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def evaluate_model(args):
    api_key, model_type, model_name, validation_data = args
    client = OpenAI(api_key=api_key)
    
    correct_count = 0
    total_count = len(validation_data)

    for example in validation_data:
        pgn = example['prompt']
        correct_response = example['completion'].strip()

        if model_type == "base":
            response = client.completions.create(
                model=model_name,
                prompt=pgn,
                temperature=0,
                max_tokens=5,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            predicted_response = response.choices[0].text.strip().split(" ")[0]
        elif model_type == "chat":
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
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt.format(pgn=pgn)}
                ],
                temperature=0,
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            predicted_response = response.choices[0].message.content.strip().split("\nAnswer: ")[-1]

        if predicted_response in correct_response:
            correct_count += 1

        print(f"Model: {model_name}, Prediction: {predicted_response} [{'Correct' if predicted_response in correct_response else 'Incorrect'}]")

    accuracy = correct_count / total_count
    print(f"\nModel: {model_name}")
    print(f"Correct predictions: {correct_count}")
    print(f"Total examples: {total_count}")
    print(f"Accuracy: {accuracy:.2%}")

    return model_name, accuracy

def main():
    api_key = "sk-proj-lH7Xxi3qCfzcueU9kutQT3BlbkFJGAC1pBw3kr3o0K1J9bES"

    models = [
        # ("chat", "gpt-4o-mini"),
        # ("chat", "ft:gpt-4o-mini-2024-07-18:devpy:puzzlegod-it1-129ex:9zLUVZ9j"),
        # ("chat", "gpt-4o"),
        # ("chat", "ft:gpt-4o-2024-08-06:devpy:puzzlegod-100ex:9zKwkawo"),
        # ("base", "babbage-002"),
        # ("base", "davinci-002"),
        ("base", "ft:davinci-002:devpy:davinci-chess-pgn:9rGcgkI9")
        # ("chat", "ft:gpt-4o-mini-2024-07-18:devpy:puzzlegod-100ex:9zKtUCDG"),
    ]

    validation_file = "chess_finetuning_val.jsonl"
    validation_data = load_validation_data(validation_file)

    # Determine the number of CPU cores to use
    num_cores = multiprocessing.cpu_count() * 8
    print(f"Using {num_cores} CPU cores")

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Prepare arguments for each model evaluation
        args = [(api_key, model_type, model_name, validation_data) for model_type, model_name in models]
        
        # Submit all tasks at once
        results = pool.map_async(evaluate_model, args)
        
        # Wait for all tasks to complete and get the results
        all_results = results.get()

    # Aggregate and print results
    print("\n| Model | Accuracy |")
    print("|-------|----------|")
    for model_name, accuracy in all_results:
        print(f"| {model_name} | {accuracy:.2%} |")

if __name__ == "__main__":
    main()