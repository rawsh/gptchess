import json
import random

def sample_jsonl(input_file, output_file, n):
    # Read all lines from the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Get the total number of examples
    total_examples = len(lines)
    
    # Ensure n is not greater than the total number of examples
    n = min(n, total_examples)
    
    # Randomly sample n indices
    sampled_indices = random.sample(range(total_examples), n)
    
    # Write the sampled examples to the output file
    with open(output_file, 'w') as f:
        for index in sampled_indices:
            f.write(lines[index])

# Example usage
input_file = 'chess_finetuning_train.jsonl'
output_file = 'chess_finetuning_train_sample.jsonl'
n = 200  # Number of examples to sample

sample_jsonl(input_file, output_file, n)
print(f"Sampled {n} examples from {input_file} and saved to {output_file}")