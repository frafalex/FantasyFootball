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

    # Create a list of players and fill it with a for loop
    players = []
    for _, row in df.iterrows():
        player_obj = player(
            position = str(row.get('R', '')).strip(),
            position_mantra = str(row.get('Rm', '')).strip(),
            name = str(row.get('Nome', '')).strip(),
            team = str(row.get('Squadra', '')).strip(),
            games_played = int(row.get('Pv', 0)),
            average_mark = float(row.get('Mv', 0.0)),
            average_fantamark = float(row.get('Fm', 0.0)),
            goals = int(row.get('Gf', 0)),
            goals_conceded = int(row.get('Gs', 0)),
            penalties_saved = int(row.get('Rp', 0)),
            penalties_taken = int(row.get('Rc', 0)),
            penalties_scored = int(row.get('R+', 0)),
            penalties_missed = int(row.get('R-', 0)),
            assists = int(row.get('Ass', 0)),
            yellow_cards = int(row.get('Amm', 0)),
            red_cards = int(row.get('Esp', 0)),
            own_goals = int(row.get('Au', 0))
        )
        players.append(player_obj)  # Append new element to the list of players

    return players


# MAIN
if __name__ == "__main__":
    
    # Folder where the data is stored
    folder_name = "Data"

    # Dictionary to store multiple seasons
    database = {}

    # Check if the folder exists
    if os.path.exists(folder_name):

        # Loop through all the files in the folder
        for name_file in os.listdir(folder_name):

            # Check if the file is valid
            if name_file.endswith('.csv'):

                file_path = os.path.join(folder_name, name_file)
                season_name = name_file.replace('Statistiche_Fantacalcio_Stagione_', '').replace('.csv', '')

                # Import the data and store it in the database
                database[season_name] = importer_parser(file_path)

    else:
        raise FileNotFoundError(f"The folder {folder_name} does not exist.")    # If the folder does not exist, raise an error