import csv
import json
import zstandard as zstd
import argparse
import tiktoken
import random
import requests
import chess

def extract_clean_pgn(game_url_with_move, game_num_mod=0):
    # Remove '/black' from the URL if present
    game_url = game_url_with_move.replace('/black', '').replace('/white', '')

    # Extract game ID and move number from the URL
    url_parts = game_url.split('/')
    game_id = url_parts[3].split('#')[0]
    
    move_number = int(url_parts[3].split('#')[1]) if '#' in url_parts[3] else None
    print(move_number)
    
    # Fetch game data from Lichess API
    api_url = f"https://lichess.org/game/export/{game_id}"
    headers = {'Accept': 'application/json'}
    response = requests.get(api_url, headers=headers)
    
    if response.status_code != 200:
        return f"Error: Unable to fetch game data. Status code: {response.status_code}", None
    
    game_data = response.json()
    moves = game_data.get('moves', '').split()

    # If move_number is None, use all moves
    if move_number is None:
        move_number = len(moves)
    else:
        # Always include the move specified in the URL
        move_number = min(move_number + game_num_mod + 1, len(moves))

    # Reconstruct the clean PGN up to the specified move
    clean_pgn = ""
    for i in range(0, move_number):
        if i % 2 == 0:
            clean_pgn += f"{i//2 + 1}. {moves[i]} "
        else:
            clean_pgn += f"{moves[i]} "

    if move_number % 2 == 0:
        clean_pgn += f"{move_number//2 + 1}. "
    
    # Trim any trailing space
    clean_pgn = clean_pgn.strip()
    
    return clean_pgn

# def create_jsonl_entry(pgn, correct_response):
#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": f"{pgn} "
#             },
#             {
#                 "role": "assistant",
#                 "content": correct_response
#             }
#         ]
#     }

# def create_jsonl_entry(pgn, correct_response):
#     return {
#         "input": pgn,
#         "output": " " + correct_response
#     }

def create_jsonl_entry(pgn, correct_response):
    return {
        "prompt": pgn,
        "completion": " " + correct_response
    }

def count_tokens(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def convert_puzzles(input_file, train_output, val_output, max_entries=None, min_rating=None, verbose=False, start_index=0):
    total_tokens_train = 0
    total_tokens_val = 0
    entries_written_train = 0
    entries_written_val = 0
    puzzles_processed = 0
    puzzles_filtered = 0
    train_entries = []
    total_rating_train = 0
    total_rating_val = 0

    with zstd.open(input_file, 'rt') as zstd_file, \
         open(val_output, 'w') as val_file:
        
        reader = csv.DictReader(zstd_file)
        
        # Skip rows until start_index
        for _ in range(start_index):
            next(reader, None)
        
        for row in reader:
            puzzle_rating = int(row['Rating'])
            if min_rating and puzzle_rating < min_rating:
                puzzles_filtered += 1
                continue

            puzzles_processed += 1
            if max_entries and puzzles_processed > max_entries:
                break

            puzzle_id = row['PuzzleId']
            game_url = row['GameUrl']
            fen = row["FEN"]
            moves = row['Moves'].split()

            moves_san = []
            board = chess.Board(fen)
            for move_idx in range(len(moves) // 2):
                opponent_move = moves[0 + move_idx * 2]
                correct_response = moves[1 + move_idx * 2]
                moves_san.append(board.san_and_push(chess.Move.from_uci(opponent_move)))
                moves_san.append(board.san_and_push(chess.Move.from_uci(correct_response)))
                
            if verbose:
                print(f"\nProcessing puzzle {puzzles_processed}")
                print(f"PuzzleId: {puzzle_id}")
                print(f"Rating: {puzzle_rating}")
                print(f"GameUrl: {game_url}")
                print(f"Moves UCI: {moves}")
                print(f"Moves SAN: {moves_san}")

            # Extract clean PGN up to the puzzle start
            game_pgn = extract_clean_pgn(game_url, -1)

            for move_idx in range(len(moves) // 2):
                opponent_move = moves_san[0 + move_idx * 2]
                correct_response = moves_san[1 + move_idx * 2]

                # Correct player to move calculation
                player_to_move = 'White' if '#' in game_url and int(game_url.split('#')[1]) % 2 == 1 else 'Black'
                if verbose:
                    print(f"Opponent's Move: {opponent_move}")
                    print(f"Correct Response: {correct_response}")
                    print(f"Player to Move: {player_to_move}")
                
                print(game_url.split('#'))
                move_num = int(game_url.split('#')[1])
                mid_mod = f" {str(move_num // 2 + move_idx + 2)}." if move_num % 2 == 1 else ""
                print(mid_mod)
                game_pgn = f"{game_pgn} {opponent_move}{mid_mod}"
                puzzle_pgn = game_pgn
                if verbose:
                    print(f"PGN: {puzzle_pgn}")

                end_mod = f" {str(move_num // 2 + move_idx + 2)}." if move_num % 2 == 0 else ""
                game_pgn = f"{game_pgn} {correct_response}{end_mod}"
                
                entry = create_jsonl_entry(puzzle_pgn, correct_response)
                entry['rating'] = puzzle_rating  # Add rating for sorting purposes
                # tokens = count_tokens(" ".join([msg["content"] for msg in entry["messages"]]))
                # tokens = count_tokens(entry["input"] + " " + entry["output"])
                tokens = count_tokens(entry["prompt"] + " " + entry["completion"])
                
                # Decide if this entry goes to train or validation set
                is_train = random.random() < 0.9
                
                if is_train:
                    train_entries.append(entry)
                    total_tokens_train += tokens
                    entries_written_train += 1
                    total_rating_train += puzzle_rating
                else:
                    del entry['rating']  # Remove rating before writing to file
                    val_file.write(json.dumps(entry) + '\n')
                    total_tokens_val += tokens
                    entries_written_val += 1
                    total_rating_val += puzzle_rating
                
                if verbose:
                    print(f"Writing entry to {'train' if is_train else 'validation'} set")
                
            if puzzles_processed % 100 == 0:
                print(f"Processed {puzzles_processed} puzzles")

    # Sort train entries by rating and write to file
    train_entries.sort(key=lambda x: x['rating'])
    with open(train_output, 'w') as train_file:
        for entry in train_entries:
            del entry['rating']  # Remove rating before writing to file
            train_file.write(json.dumps(entry) + '\n')

    avg_rating_train = total_rating_train / entries_written_train if entries_written_train else 0
    avg_rating_val = total_rating_val / entries_written_val if entries_written_val else 0

    print(f"Processed {puzzles_processed} puzzles")
    print(f"Filtered {puzzles_filtered} puzzles")
    print(f"Written {entries_written_train} entries to train set")
    print(f"Written {entries_written_val} entries to validation set")
    print(f"Total tokens in train set: {total_tokens_train}")
    print(f"Total tokens in validation set: {total_tokens_val}")
    print(f"Average tokens per entry (train): {total_tokens_train / entries_written_train if entries_written_train else 0:.2f}")
    print(f"Average tokens per entry (validation): {total_tokens_val / entries_written_val if entries_written_val else 0:.2f}")
    print(f"Average rating (train): {avg_rating_train:.2f}")
    print(f"Average rating (validation): {avg_rating_val:.2f}")
    print("Conversion complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chess puzzles to JSONL format with PGN, sorted train set, train/validation split, optional HTML visualization, token counting, and rating filter")
    parser.add_argument("--input_file", default="lichess_db_puzzle.csv.zst", help="Input file path (default: lichess_db_puzzle.csv.zst)")
    parser.add_argument("--train_output", default="chess_finetuning_train.jsonl", help="Output JSONL file path for training set (default: chess_finetuning_train.jsonl)")
    parser.add_argument("--val_output", default="chess_finetuning_val.jsonl", help="Output JSONL file path for validation set (default: chess_finetuning_val.jsonl)")
    parser.add_argument("--max_entries", type=int, default=10, help="Maximum number of puzzles to process (default: 10, use 0 for all puzzles)")
    parser.add_argument("--min_rating", type=int, help="Minimum puzzle rating to include (default: None)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of each example")
    parser.add_argument("--start_index", type=int, default=0, help="Start processing from this index in the CSV file (default: 0)")
    
    args = parser.parse_args()
    max_entries = None if args.max_entries == 0 else args.max_entries

    convert_puzzles(args.input_file, args.train_output, args.val_output, max_entries, args.min_rating, args.verbose, args.start_index)