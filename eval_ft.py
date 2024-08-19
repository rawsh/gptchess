import json
from openai import OpenAI

def load_validation_data(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def evaluate_model(client, model_name, validation_data):
    correct_count = 0
    total_count = len(validation_data)

    for example in validation_data:
        pgn = example['prompt']
        correct_response = example['completion'].strip()

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

        if predicted_response in correct_response:
            print(f"Prediction: {predicted_response} [Correct]")
            correct_count += 1
        else:
            print(f"Prediction: {predicted_response} [Incorrect], Ground Truth: {correct_response}")

    accuracy = correct_count / total_count
    return correct_count, total_count, accuracy

def evaluate_chat_model(client, model_name, validation_data):
    correct_count = 0
    total_count = len(validation_data)

    for example in validation_data:
        pgn = example['prompt']
        correct_response = example['completion'].strip()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a chess puzzle expert. Never include the move number. Only respond with the correct next move."},
                {"role": "user", "content": pgn}
            ],
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        predicted_response = response.choices[0].message.content.strip().split(" ")[0]

        if predicted_response in correct_response:
            print(f"Prediction: {predicted_response} [Correct]")
            correct_count += 1
        else:
            print(f"Prediction: {predicted_response} [Incorrect], Ground Truth: {correct_response}")

    accuracy = correct_count / total_count
    return correct_count, total_count, accuracy

def main():
    api_key = "sk-proj-lH7Xxi3qCfzcueU9kutQT3BlbkFJGAC1pBw3kr3o0K1J9bES"


    # model_name = "ft:davinci-002:devpy:davinci-chess-pgn:9rBeLKu5"
    # model_name = "ft:davinci-002:devpy:davinci-chess-pgn:9rCcxRKP"
    # model_name = "ft:davinci-002:devpy:davinci-chess-pgn:9rCw0QT6"
    # model_name = "ft:davinci-002:devpy:davinci-chess-pgn:9rDEGCk6"
    # model_name = "ft:davinci-002:devpy:davinci-chess-pgn:9rGcgkI9"
    
    # model_name = "gpt-3.5-turbo-instruct"
    # model_name = "davinci-002"
    # model_name = "babbage-002"
    # validation_file = "chess_finetuning_val.jsonl"
    # validation_file = "chess_finetuning_train.jsonl"

    # all elo
    # babbage: Accuracy: 54.42%
    # davinci: Accuracy: 74.19%
    # davinci ft 1: Accuracy: 75.35%
    # davinci ft 5: Accuracy: 80.82%

    # 2000+ elo
    # davinci: 61.90%
    # davinci ft 1: 61.90%
    # davinci ft 2: 57.14%
    # davinci ft 3: 58.73%
    # davinci ft 4: 60.32%
    # davinci ft 5: 65.08%

    client = OpenAI(api_key=api_key)
    # validation_data = load_validation_data(validation_file)

    # correct, total, accuracy = evaluate_model(client, model_name, validation_data)

    # print(f"Correct predictions: {correct}")
    # print(f"Total examples: {total}")
    # print(f"Accuracy: {accuracy:.2%}")

    models = [
        ("chat", "gpt-3.5-turbo"),
        ("chat", "gpt-4o-mini"),
        # ("base", "davinci-002"),
        # ("base", "babbage-002"),
        # ("base", "ft:davinci-002:devpy:davinci-chess-pgn:9rGcgkI9")
        # Add more models as needed
    ]

    validation_file = "chess_finetuning_val.jsonl"
    validation_data = load_validation_data(validation_file)

    results = {}

    for model_info in models:
        model_type, model_name = model_info
        print(f"\nEvaluating {model_type} model: {model_name}")
        if model_type == "base":
            correct, total, accuracy = evaluate_model(client, model_name, validation_data)
        elif model_type == "chat":
            correct, total, accuracy = evaluate_chat_model(client, model_name, validation_data)
        results[model_name] = accuracy
        print(f"Correct predictions: {correct}")
        print(f"Total examples: {total}")
        print(f"Accuracy: {accuracy:.2%}")

    # Print summary table
    print("\n| Model | Accuracy |")
    print("|-------|----------|")
    for model_name, accuracy in results.items():
        print(f"| {model_name} | {accuracy:.2%} |")

if __name__ == "__main__":
    main()