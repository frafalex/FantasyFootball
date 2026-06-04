"""
----------------------------------------------------------------------
statistical_analysis.py:
This script performs statistical analysis on the imported player data.
----------------------------------------------------------------------
"""


# PACKS
import os   # For file handling
import numpy as np      # For numerical operations
import pandas as pd     # For data analysis
import scipy.stats as stats     # For statistical tests
import matplotlib.pyplot as plt         # For data visualization
from dataclasses import asdict          # For converting dataclass objects to dictionaries
from dataclasses import dataclass       # For creating data structures
from functions.plots import plot_dist, kde_plt, gauss_plt, gamma_plt    # For plotting data distribution

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
            name = str(row.get('Nome', '')).strip(),
            position = str(row.get('R', '')).strip(),
            position_mantra = str(row.get('Rm', '')).strip(),
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

def position_analysis_classic(database: dict, plots: bool):
    """
    This function performs statistical analysis on the player data devided in groups according to their positions in classic mode.

    Args:
        database (dict): A dictionary containing player data
        plots (bool): Whether to generate plots.

    Returns:
        df: A pandas DataFrame containing the statistical data.
    """
    # Conversion of data into a pandas DataFrame
    flat_data = []
    for season, players in database.items():
        for p in players:

            # Convert dataset in a dictionary
            player_dict = asdict(p)
            player_dict['season'] = season
            flat_data.append(player_dict)

    # Create the DataFrame (this is like an excel table with columns: index, season, name, position, ...)
    df = pd.DataFrame(flat_data)

    # Number of players per position for each season and average
    annual_counts = df.groupby(['season', 'position']).size().reset_index(name='count')
    average_number = annual_counts.groupby('position')['count'].mean()
    number_std = annual_counts.groupby('position')['count'].std()

    # DataFrame with only active players (those who played at least one game)
    df_active = df[df['games_played'] > 0].copy()

    # Statistics normalized by games played
    df_active['goals_per_game'] = df_active['goals'] / df_active['games_played']
    df_active['assists_per_game'] = df_active['assists'] / df_active['games_played']
    df_active['yellows_per_game'] = df_active['yellow_cards'] / df_active['games_played']
    df_active['reds_per_game'] = df_active['red_cards'] / df_active['games_played']
    df_active['own_goals_per_game'] = df_active['own_goals'] / df_active['games_played']
    df_active['goals_conceded_per_game'] = df_active['goals_conceded'] / df_active['games_played']
    df_active['penalties_saved_per_game'] = df_active['penalties_saved'] / df_active['games_played']

    # Penalties statisics
    pen_takers = df_active[df_active['penalties_taken'] > 0].copy()
    pen_takers_number = pen_takers.groupby(['season', 'position']).size().reset_index(name='count')
    pen_takers_average = pen_takers_number.groupby('position')['count'].mean()
    pen_takers_std = pen_takers_number.groupby('position')['count'].std()

    if plots:

        # Assists per game (0-0.5, 20 bins means 0.025 per bin)
        plot_dist(df_active, 'assists_per_game', 'position', 'Normalized Assists', num_bins=20, max_x=0.5)

        # Yellow cards per game (0-0.5, 20 bins means 0.025 per bin)
        plot_dist(df_active, 'yellows_per_game', 'position', 'Normalized Yellow Cards', num_bins=20, max_x=0.5)

        # Red cards per game (0-0.1, 20 bins means 0.005 per bin)
        plot_dist(df_active, 'reds_per_game', 'position', 'Normalized Red Cards', num_bins=20, max_x=0.1)

        # Own goals per game (0-0.1, 20 bins means 0.005 per bin)
        plot_dist(df_active, 'own_goals_per_game', 'position', 'Normalized Own Goals', num_bins=20, max_x=0.1)

        # Penalties saved per game (0-0.1, 20 bins means 0.005 per bin)
        plot_dist(df_active, 'penalties_saved_per_game', 'position', 'Normalized Penalties Saved', num_bins=20, max_x=0.1)

        # Goals conceded per game (0-3.0, 30 bins means 0.1 per bin)
        plot_dist(df_active, 'goals_conceded_per_game', 'position', 'Normalized Goals Conceded', num_bins=30, max_x=3.5)

    # Statistical analysis per position
    dna_players = {}
    positions = df['position'].unique()
    
    dna_players = {}
    positions = df['position'].unique()
    
    for pos in positions:
        dna_players[pos] = {}

        # --- GAMES PLAYED ANALYSIS ---
        games_played = df[df['position'] == pos]['games_played']

        # Probability of zero games played
        prob_zero_games = len(games_played[games_played == 0]) / len(games_played)
        dna_players[pos]['prob_zero'] = prob_zero_games

        # KDE fit on games played (only active players)
        games_played_active = df_active[df_active['position'] == pos]['games_played']

        # Mirroring technique to handle the boundary at zero for KDE fitting
        mirrored_data = 2 - games_played_active
        extended_data = np.concatenate([games_played_active, mirrored_data])
        
        # Role-specific bandwidth selection for KDE fitting
        if pos == 'P':
            bw = 0.08   # Sharper distribution for goalkeepers
        else:
            bw = 0.15   # Smoother distribution for outfield players

        # Generate and store the KDE model in the dna_players dictionary
        dna_players[pos]['kde_gp'] = stats.gaussian_kde(extended_data, bw_method=bw)

        # --- AVERAGE MARK ANALYSIS ---
        avg_mark = df_active[df_active['position'] == pos]['average_mark']

        # Standard Gaussian fit for average mark
        dna_players[pos]['mean_mark'] = avg_mark.mean()
        dna_players[pos]['std_mark'] = avg_mark.std()

        # --- NORMALIZED GOALS ANALYSIS ---
        goals_per_game = df_active[df_active['position'] == pos]['goals_per_game']
        
        # Calculate the probability of exactly zero goals
        prob_zero_goals = len(goals_per_game[goals_per_game == 0]) / len(goals_per_game)
        dna_players[pos]['prob_zero_goals'] = prob_zero_goals
        
        # Extract only strictly positive rates for the Gamma fit
        positive_goals = goals_per_game[goals_per_game > 0]
        
        # Fit the Gamma distribution
        if len(positive_goals) > 0:
            shape, loc, scale = stats.gamma.fit(positive_goals, floc=0)
        else:
            # Fallback for positions with literally zero goals (e.g., some seasons for GKs)
            shape, loc, scale = 0, 0, 0
            
        # Store parameters in the dictionary
        dna_players[pos]['gamma_shape_goals_per_game'] = shape
        dna_players[pos]['gamma_scale_goals_per_game'] = scale

    if plots:
        kde_plt(df_active, dna_players)
        gauss_plt(df_active, dna_players)
        gamma_plt(df_active, dna_players, 'goals_per_game', 'Gamma Fit for Normalized Goals')

    return df


# MAIN
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

# Perform the analysis
plots = True
df = position_analysis_classic(database, plots)

# Plot display
if plots:
    plt.show()
