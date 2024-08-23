import json
import tiktoken
import argparse

def count_tokens(filename):
    # Initialize the GPT-4 tokenizer
    enc = tiktoken.encoding_for_model("gpt-4")
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            try:
                data = json.loads(line)
                prompt = data["messages"][0]["content"]
                completion = data["messages"][1]["content"]
                
                prompt_tokens = len(enc.encode(prompt))
                completion_tokens = len(enc.encode(completion))
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                
                print(f"Line {line_number} - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON on line {line_number}: {line}")
    
    print(f"\nTotal prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count tokens in a JSONL file using GPT-4 tokenizer")
    parser.add_argument("filename", help="Path to the JSONL file")
    
    args = parser.parse_args()
    
    count_tokens(args.filename)