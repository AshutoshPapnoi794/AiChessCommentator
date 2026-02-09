import requests

def fetch_pgn(username, max_games=10):
    """
    Fetches PGN of games for a given Lichess username, including opening info.
    
    :param username: Lichess username
    :param max_games: Number of games to fetch (default: 10)
    :return: PGN string
    """
    url = f"https://lichess.org/api/games/user/{username}"

    headers = {
        "Accept": "application/x-chess-pgn"
    }

    params = {
        "max": max_games,       # Limit games (remove if you want all)
        "moves": True,
        "pgnInJson": False,
        "evals": False,
        "opening": True         # ✅ Add opening info
    }

    response = requests.get(url, headers=headers, params=params, stream=True)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching PGN: {response.status_code}")
        return None


if __name__ == "__main__":
    username = input("Enter Lichess username: ").strip()
    pgn_data = fetch_pgn(username, max_games=20)  # You can change max_games

    if pgn_data:
        print("\n===== PGN DATA (with openings) =====\n")
        print(pgn_data)
