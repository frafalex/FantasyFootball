"""
----------------------------------------------------------------------
statistical_analysis.py:
This script performs statistical analysis on the imported player data.
----------------------------------------------------------------------
"""


# PACKS
from ast import For
import os   # For file handling
import numpy as np      # For numerical operations
import pandas as pd     # For data analysis
import scipy.stats as stats     # For statistical tests
import matplotlib.pyplot as plt         # For data visualization
from dataclasses import asdict          # For converting dataclass objects to dictionaries
from dataclasses import dataclass       # For creating data structures
from functions.plots import kde_plt, gauss_plt, gamma_plt, beta_plt     # For plotting data distributions

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

def fit_zero_inflated_gamma(data: pd.Series) -> tuple:
    """
    Calculates the probability of zero and fits a Gamma distribution to strictly positive values.

    Args:
        data (pd.Series): The data series to analyze (e.g., normalized goals, red cards).

    Returns:
        tuple: (probability of zero, gamma shape parameter, gamma scale parameter).
    """
    # Calculate the probability of exactly zero
    if len(data) > 0:
        prob_zero = len(data[data == 0]) / len(data)
    else:
        prob_zero = 1.0

    # Extract only strictly positive rates for the Gamma fit
    positive_data = data[data > 0]

    # Fit the Gamma distribution
    if len(positive_data) > 0:
        shape, _, scale = stats.gamma.fit(positive_data, floc=0)
    else:
        # Fallback for positions with literally zero events
        shape, scale = 0.0, 0.0

    return prob_zero, shape, scale

def position_analysis_classic(database: dict, plots: bool):
    """
    This function performs statistical analysis on the player data devided in groups according to their positions in classic mode.

    Args:
        database (dict): A dictionary containing player data
        plots (bool): Whether to generate plots.

    Returns:
        df: A pandas DataFrame containing the statistical data;
        dna_players: A dictionary containing the parameters of the fitted distributions for each position;
        average_number: A pandas Series containing the average number of players per position;
        number_std: A pandas Series containing the standard deviation of the number of players per position.
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


        # --- GAMMA FIT ANALYSIS FOR NORMALIZED METRICS ---
        active_pos = df_active[df_active['position'] == pos]

        # Define the normalized metrics to analyze and their corresponding dictionary suffixes.
        gamma_metrics = {
            'goals_per_game': 'goals',
            'assists_per_game': 'assists',
            'yellows_per_game': 'yellow_cards',
            'reds_per_game': 'red_cards',
            'own_goals_per_game': 'own_goals'
        }

        # Add Goalkeeper-specific metrics dynamically
        if pos == 'P':
            gamma_metrics['goals_conceded_per_game'] = 'goals_conceded'
            gamma_metrics['penalties_saved_per_game'] = 'penalties_saved'

        # Loop through each metric, extract parameters, and store them in the DNA dictionary
        for col_name, suffix in gamma_metrics.items():
            
            # Call the custom function
            prob_zero, shape, scale = fit_zero_inflated_gamma(active_pos[col_name])
            
            # Store the three parameters dynamically
            dna_players[pos][f'prob_zero_{suffix}'] = prob_zero
            dna_players[pos][f'gamma_shape_{suffix}'] = shape
            dna_players[pos][f'gamma_scale_{suffix}'] = scale


        # --- PENALTY TAKERS ANALYSIS ---
        pen_takers = active_pos['penalties_taken'] > 0

        # Probability of being a designated penalty taker
        prob_is_penalty_taker = pen_takers.sum() / len(active_pos) if len(active_pos) > 0 else 0
        dna_players[pos]['prob_is_penalty_taker'] = prob_is_penalty_taker
        
        # Extract only the players who actually took at least one penalty
        actual_pen_takers = active_pos[pen_takers].copy()
        
        if len(actual_pen_takers) > 0:
            # Frequency: How many penalties do they take per game?
            pen_taken_per_game = actual_pen_takers['penalties_taken'] / actual_pen_takers['games_played']
            shape_pen, _, scale_pen = stats.gamma.fit(pen_taken_per_game, floc=0)
            
            # Conversion Rate: How many do they score?
            conversion_rate = actual_pen_takers['penalties_scored'] / actual_pen_takers['penalties_taken']

            # We clip exact 0s and 1s to mathematically acceptable approximations (e.g., 0.9999).
            conversion_rate_clipped = np.clip(conversion_rate, 0.0001, 0.9999)
            
            # Fit the Beta distribution using the clipped data
            a_beta, b_beta, _, _ = stats.beta.fit(conversion_rate_clipped, floc=0, fscale=1)
            
        else:
            # Fallback for positions with zero penalty takers
            shape_pen, scale_pen = 0.0, 0.0
            a_beta, b_beta = 1.0, 1.0  # Default uniform distribution parameters
            
        # Store parameters in the DNA dictionary
        dna_players[pos]['gamma_shape_penalties_taken'] = shape_pen
        dna_players[pos]['gamma_scale_penalties_taken'] = scale_pen
        dna_players[pos]['beta_a_pen_conversion'] = a_beta
        dna_players[pos]['beta_b_pen_conversion'] = b_beta
        
    if plots:
        kde_plt(df_active, dna_players)
        gauss_plt(df_active, dna_players)
        gamma_plt(df_active, dna_players, 'goals_per_game', 'goals')
        gamma_plt(df_active, dna_players, 'assists_per_game', 'assists')
        gamma_plt(df_active, dna_players, 'yellows_per_game', 'yellow_cards')
        gamma_plt(df_active, dna_players, 'reds_per_game', 'red_cards')
        gamma_plt(df_active, dna_players, 'own_goals_per_game', 'own_goals')
        gamma_plt(df_active, dna_players, 'goals_conceded_per_game', 'goals_conceded')
        gamma_plt(df_active, dna_players, 'penalties_saved_per_game', 'penalties_saved')
        beta_plt(df_active, dna_players)

    return df, dna_players, average_number, number_std


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
