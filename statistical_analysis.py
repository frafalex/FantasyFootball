"""
----------------------------------------------------------------------
statistical_analysis.py:
This script performs statistical analysis on the imported player data.
----------------------------------------------------------------------
"""

# PACKS
import os   # for file handling
import pandas as pd   # for data analysis
from dataclasses import dataclass   # for creating data structures

# CLASSES
@dataclass
class player:
    name: str
    position: str
    position_mantra: str
    team: str
    games_played: int
    average_mark: float
    average_fantamark: float
    goals: int
    goals_conceded: int
    penalties_saved: int
    penalties_taken: int
    penalties_scored: int
    penalties_missed: int
    assists: int
    yellow_cards: int
    red_cards: int
    own_goals: int

# FUNCTIONS
def importer_parser(file_path: str) -> list[player]:
    """
    This function imports player data from a .csv file and returns a list of player objects.

    Args:
        file_path (str): The path to the file containing player data.

    Returns:
        list[player]: A list of player objects.
    """

    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Import the data using pandas
    df = pd.read_csv(file_path, sep=';', decimal=',')

    # Remove eventual leading and trailing whitespaces
    df.columns = df.columns.str.strip()

    # Create a list of players
    players = []
    for index, row in df.iterrows():
        player_obj = player(
            position = str(row.get('Ruolo', '')).strip(),
            position_mantra = str(row.get('Ruolo mantra', '')).strip(),
            name = str(row.get('Nome', '')).strip(),
            team = str(row.get('Squadra', '')).strip(),
            games_played = int(row.get('Partite a voto', 0)),
            average_mark = float(row.get('Media voto', 0.0)),
            average_fantamark = float(row.get('Fanta media', 0.0)),
            goals = int(row.get('Gol fatti', 0)),
            goals_conceded = int(row.get('Gol subiti', 0)),
            penalties_saved = int(row.get('Rigori parati', 0)),
            penalties_taken = int(row.get('Rigori calciati', 0)),
            penalties_scored = int(row.get('Rigori segnati', 0)),
            penalties_missed = int(row.get('Rigori sbagliati', 0)),
            assists = int(row.get('Assist', 0)),
            yellow_cards = int(row.get('Ammonizioni', 0)),
            red_cards = int(row.get('Espulsioni', 0)),
            own_goals = int(row.get('Autogol', 0))
        )
        players.append(player_obj)  # Append the player object to the list of players

    return players

# MAIN
if __name__ == "__main__":
    # Example usage
    file_path = 'Data/Statistiche_Fantacalcio_Stagione_2025_26.csv'
    players_list = importer_parser(file_path)
    print(players_list[156].name)