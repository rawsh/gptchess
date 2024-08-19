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

        messages = [
            {"role": "user", "content": pgn + " "}
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=10,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        predicted_response = response.choices[0].message.content.strip().split()[0]

        if predicted_response in correct_response:
            print(f"Prediction: {predicted_response} [Correct]")
            correct_count += 1
        else:
            print(f"Prediction: {predicted_response} [Incorrect], Ground Truth: {correct_response}")

    accuracy = correct_count / total_count
    return correct_count, total_count, accuracy

def main():
    api_key = "sk-proj-lH7Xxi3qCfzcueU9kutQT3BlbkFJGAC1pBw3kr3o0K1J9bES"
    model_name = "ft:gpt-4o-mini-2024-07-18:devpy:chess-100-pgn-3:9rBr7IBM"
    validation_file = "chess_finetuning_val.jsonl"

    client = OpenAI(api_key=api_key)
    validation_data = load_validation_data(validation_file)

    correct, total, accuracy = evaluate_model(client, model_name, validation_data)

    print(f"Correct predictions: {correct}")
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()