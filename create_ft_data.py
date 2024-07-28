import csv
import json
import zstandard as zstd
import chess
import chess.pgn
import chess.svg
import argparse
import tiktoken
import random
import io

def create_jsonl_entry(pgn, correct_move, rating):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a chess engine. Given a chess position in PGN (Portable Game Notation) format, your task is to provide the best move in UCI (Universal Chess Interface) notation."
            },
            {
                "role": "user",
                "content": f"What is the best move for this chess position? PGN:\n{pgn}"
            },
            {
                "role": "assistant",
                "content": correct_move
            }
        ],
        "rating": rating
    }

def count_tokens(text, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def convert_puzzles(input_file, train_output, val_output, html_output=None, max_entries=None, min_rating=None, verbose=False):
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
        
        html_file = open(html_output, 'w') if html_output else None
        
        if html_file:
            html_file.write('''
            <html>
            <head>
                <style>
                    .puzzle { margin-bottom: 30px; border: 1px solid #ccc; padding: 10px; }
                    .board { display: inline-block; margin-right: 20px; }
                    .move { margin-bottom: 10px; }
                </style>
            </head>
            <body>
            ''')
        
        for row in reader:
            puzzle_rating = int(row['Rating'])
            if min_rating and puzzle_rating < min_rating:
                puzzles_filtered += 1
                continue

            puzzles_processed += 1
            if max_entries and puzzles_processed > max_entries:
                break

            puzzle_id = row['PuzzleId']
            initial_fen = row['FEN']
            moves = row['Moves'].split()
            game_url = row['GameUrl']
            
            if verbose:
                print(f"\nProcessing puzzle {puzzles_processed}")
                print(f"PuzzleId: {puzzle_id}")
                print(f"Rating: {puzzle_rating}")
                print(f"Initial FEN: {initial_fen}")
                print(f"Moves: {moves}")
                print(f"GameUrl: {game_url}")
            
            if html_file:
                html_file.write(f'<div class="puzzle"><h2>Puzzle {puzzles_processed}</h2>')
                html_file.write(f'<p>PuzzleId: {puzzle_id}</p>')
                html_file.write(f'<p>Rating: {puzzle_rating}</p>')
                html_file.write(f'<p>Initial FEN: {initial_fen}</p>')
                html_file.write(f'<p>Moves: {" ".join(moves)}</p>')
                html_file.write(f'<p>GameUrl: <a href="{game_url}" target="_blank">{game_url}</a></p>')
            
            # Create a game from the initial FEN
            game = chess.pgn.Game()
            game.setup(initial_fen)
            node = game
            
            # Apply all moves and create PGN
            for i, move in enumerate(moves):
                node = node.add_variation(chess.Move.from_uci(move))
                current_player = 'White' if i % 2 == 0 else 'Black'
                
                # Create PGN for the current position
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                pgn = game.accept(exporter)
                
                # Only create entries for the solving player's moves
                if (i % 2 == 0 and game.board().turn == chess.WHITE) or (i % 2 == 1 and game.board().turn == chess.BLACK):
                    entry = create_jsonl_entry(pgn, move, puzzle_rating)
                    json_string = json.dumps(entry)
                    tokens = count_tokens(json_string)
                    
                    # Decide if this entry goes to train or validation set
                    is_train = random.random() < 0.9
                    
                    if is_train:
                        train_entries.append(entry)
                        total_tokens_train += tokens
                        entries_written_train += 1
                        total_rating_train += puzzle_rating
                    else:
                        val_file.write(json_string + '\n')
                        total_tokens_val += tokens
                        entries_written_val += 1
                        total_rating_val += puzzle_rating
                    
                    if verbose:
                        print(f"Writing move {i+1}: {move} ({current_player}'s move) to {'train' if is_train else 'validation'} set")
                elif verbose:
                    print(f"Skipping move {i+1}: {move} (opponent's move)")
                
                if html_file:
                    svg = chess.svg.board(node.board(), size=300)
                    html_file.write(f'<div class="move">')
                    html_file.write(f'<div class="board">{svg}</div>')
                    html_file.write(f'<p>Move {i+1}: {move} ({current_player} moved)</p>')
                    html_file.write(f'<p>PGN:<br><pre>{pgn}</pre></p>')
                    html_file.write('</div>')
            
            if html_file:
                html_file.write('</div>')
            
            if puzzles_processed % 1000 == 0:
                print(f"Processed {puzzles_processed} puzzles")

        if html_file:
            html_file.write('</body></html>')
            html_file.close()

    # Sort train entries by rating and write to file
    train_entries.sort(key=lambda x: x['rating'])
    with open(train_output, 'w') as train_file:
        for entry in train_entries:
            json_string = json.dumps(entry)
            train_file.write(json_string + '\n')

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
    parser.add_argument("--html_output", default="chess_puzzles_visualization.html", help="Output HTML file path for visualization (default: chess_puzzles_visualization.html)")
    parser.add_argument("--max_entries", type=int, default=10, help="Maximum number of puzzles to process (default: 10, use 0 for all puzzles)")
    parser.add_argument("--min_rating", type=int, help="Minimum puzzle rating to include (default: None)")
    parser.add_argument("--no_html", action="store_true", help="Disable HTML visualization generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging of each example")
    
    args = parser.parse_args()

    html_output = None if args.no_html else args.html_output
    max_entries = None if args.max_entries == 0 else args.max_entries

    convert_puzzles(args.input_file, args.train_output, args.val_output, html_output, max_entries, args.min_rating, args.verbose)