import requests

def extract_clean_pgn(game_url_with_move):
    # Remove '/black' from the URL if present
    game_url = game_url_with_move.replace('/black', '')

    # Extract game ID and move number from the URL
    url_parts = game_url.split('/')
    game_id = url_parts[3].split('#')[0]
    
    move_number = int(url_parts[3].split('#')[1]) if '#' in url_parts[3] else None
    
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
        move_number = min(move_number + 1, len(moves))

    # Reconstruct the clean PGN up to the specified move
    clean_pgn = ""
    for i in range(0, move_number):
        if i % 2 == 0:
            clean_pgn += f"{i//2 + 1}. {moves[i]} "
        else:
            clean_pgn += f"{moves[i]} "
    
    # Trim any trailing space
    clean_pgn = clean_pgn.strip()
    
    return clean_pgn, move_number

# Example usage
game_url = "https://lichess.org/vaqz2bx6#36"
clean_pgn, move_number = extract_clean_pgn(game_url)
print(f"Clean PGN: {clean_pgn}")
print(f"Move number: {move_number}")