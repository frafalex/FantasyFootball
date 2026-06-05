import math     # For mathematical functions, specifically for calculating grid dimensions
import numpy as np  # For numerical operations, specifically for generating points for the Beta curve
import pandas as pd         # For data manipulation
import seaborn as sns       # For data visualization
from scipy import stats     # For statistical functions, specifically for fitting the Beta distribution
import matplotlib.pyplot as plt     # For data visualization


# Plot data distribution with optional categorization (e.g., by position)
def plot_dist(df: pd.DataFrame, target_column: str, category_column: str = None, title: str = "Distribution Plot", num_bins: int = 39, max_x: int = 38) -> None:

    """
    Generates a histogram with a Kernel Density Estimate (KDE) curve.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the player data;
        target_column (str): The name of the column for which to plot the distribution;
        category_column (str, optional): The name of the column to use for categorizing the data (e.g., 'position'). If None, no categorization is applied. Defaults to None;
        title (str, optional): The title of the plot. Defaults to "Distribution Plot";
        num_bins (int, optional): The number of bins to use in the histogram. Defaults to 39;
        max_x (int, optional): The maximum value on the x-axis. Defaults to 38.

    Returns:
        None: The function does not return anything, it just shows the plot.
    """
    # Set a clean visual theme
    sns.set_theme(style="whitegrid")
    
# CASE 1: Categorized Plot (e.g., divided by Position)
    if category_column:
        # Extract unique categories dynamically (e.g., 'P', 'D', 'C', 'A')
        categories = sorted(df[category_column].dropna().unique())
        n_categories = len(categories)
        
        # Calculate optimal grid dimensions (e.g., 2x2, 2x3)
        n_cols = math.ceil(math.sqrt(n_categories))
        n_rows = math.ceil(n_categories / n_cols)
        
        # Create subplots. sharey is False by default so each gets its own Y-axis.
        # squeeze=False ensures 'axes' is always a 2D array, preventing iteration errors.
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten the 2D array of axes for easy 1D iteration
        axes = axes.flatten()
            
        for i, category in enumerate(categories):
            subset = df[df[category_column] == category]
            
            sns.histplot(
                data=subset,
                x=target_column,
                bins=num_bins,
                binrange=(0, max_x),
                ax=axes[i],
                color=sns.color_palette("husl", n_categories)[i]
            )
            
            # Format subplot titles and labels
            axes[i].set_title(f"{category_column.capitalize()}: {category}", fontsize=14)
            axes[i].set_xlabel(target_column.replace('_', ' ').title())
            axes[i].set_ylabel("Number of Players") # Re-applied to all since Y-axis is independent
                
        # Hide any unused subplots if the grid is larger than the number of categories
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()

    # CASE 2: Single Overall Plot (No categorization)
    else:
        plt.figure(figsize=(10, 6))
        
        sns.histplot(
            data=df,
            x=target_column,
            bins=num_bins,
            binrange=(0, max_x),
            kde=True,
            color="steelblue"
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(target_column.replace('_', ' ').title())
        plt.ylabel("Number of Players")
        
        plt.tight_layout()


# Model validation plot for KDE fit on games played
def kde_plt(df: pd.DataFrame, dna_players: dict) -> None:
    """
    Generates a 2x2 subplot grid to validate the KDE fit for games played by active players in all roles.

    Args:
        df (pd.DataFrame): The DataFrame containing active player data;
        dna_players (dict): A dictionary containing the fitted KDE model for each role.
    """    
    # Define the positions to loop through
    positions = ['P', 'D', 'C', 'A']
    
    # Figure setup: 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Validation KDE Fit for Games Played", fontsize=18, fontweight='bold')
    sns.set_theme(style="whitegrid")
    
    # Flatten the 2x2 matrix into a 1D list so we can iterate easily
    axes = axes.flatten()
    
    for i, position in enumerate(positions):
        ax = axes[i] # Select the specific subplot
        
        # Extract the real data for the specified position
        games_played = df[df['position'] == position]['games_played']
        
        # Retrieve the KDE model from the dictionary
        kde_model = dna_players[position]['kde_gp']
        
        # Real histogram (38 bins, restricted from 1 to 38 to exclude zeros)
        sns.histplot(
            games_played, 
            bins=38, 
            binrange=(1, 38), 
            stat='density', 
            color='lightsteelblue', 
            label=f'Real Data ({position})',
            ax=ax # Directs the plot to the current subplot
        )
        
        # KDE curve generation (evaluating points from 1 to 38)
        x = np.linspace(1, 38, 200)
        y = kde_model.evaluate(x) * 2   # Scaling factor to adjust for the mirroring technique
        
        # KDE curve plotting
        ax.plot(x, y, color='crimson', linewidth=3, label='KDE Fit')
        
        # Plot formatting for the individual subplot
        ax.set_title(f"Position: {position}", fontsize=14)
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Probability Density")
        ax.legend()
    
    # Adjust layout to prevent overlapping text
    plt.tight_layout()


# Model validation plot for Gaussian fit on average mark
def gauss_plt(df: pd.DataFrame, dna_players: dict) -> None:
    """
    Generates a 2x2 subplot grid to validate the Gaussian fit for average marks of active players.

    Args:
        df (pd.DataFrame): The DataFrame containing active player data;
        dna_players (dict): A dictionary containing the fitted Gaussian parameters for each role.
    """    
    # Define the positions to loop through
    positions = ['P', 'D', 'C', 'A']
    
    # Figure setup: 2 rows, 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Validation Gaussian Fit for Average Mark", fontsize=18, fontweight='bold')
    sns.set_theme(style="whitegrid")
    
    # Flatten the 2x2 matrix into a 1D list so we can iterate easily
    axes = axes.flatten()
    
    for i, position in enumerate(positions):
        ax = axes[i] # Select the specific subplot
        
        # Extract the real data for the specified position
        average_marks = df[df['position'] == position]['average_mark']
        
        # Retrieve the Gaussian parameters from the dictionary
        mu = dna_players[position]['mean_mark']
        sigma = dna_players[position]['std_mark']
        
        # Real histogram (letting seaborn decide the optimal bins for the bell shape)
        sns.histplot(
            average_marks, 
            stat='density', 
            color='mediumaquamarine', 
            label=f'Real Data ({position})',
            ax=ax
        )
        
        # Gaussian curve generation
        # We draw the curve spanning exactly 4 standard deviations left and right from the mean 
        # to perfectly frame the bell shape without leaving empty spaces.
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        # Curve plotting
        ax.plot(x, y, color='crimson', linewidth=3, label=f'Gaussian Fit (μ={mu:.2f}, σ={sigma:.2f})')
        
        # Plot formatting for the individual subplot
        ax.set_title(f"Position: {position}", fontsize=14)
        ax.set_xlabel("Average Mark")
        ax.set_ylabel("Probability Density")
        ax.legend()
    
    # Adjust layout to prevent overlapping text
    plt.tight_layout()


# Model validation plot for Gamma fit on strictly positive rates
def gamma_plt(df: pd.DataFrame, dna_players: dict, column_name: str, target_name: str) -> None:
    """
    Generates a 2x2 subplot grid to validate the Gamma fit for specific rates (e.g., goals, assists).
    It filters out exact zeros to focus purely on the distribution of the positive rates.

    Args:
        df (pd.DataFrame): The DataFrame containing active player data;
        dna_players (dict): A dictionary containing the fitted Gamma parameters;
        column_name (str): The specific column to analyze (e.g., 'goals_per_game');
        target_name (str): The clean name for the plots and dict keys (e.g., 'own_goals').
    """    
    positions = ['P', 'D', 'C', 'A']
    
    # Format the target_name for display (e.g., 'own_goals' -> 'Own Goals')
    display_name = target_name.replace('_', ' ').title()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Validation Gamma Fit for strictly positive {display_name}", fontsize=18, fontweight='bold')
    sns.set_theme(style="whitegrid")
    
    axes = axes.flatten()
    
    for i, position in enumerate(positions):
        ax = axes[i]
        
        # We keep using the original target_name for the dictionary keys
        dict_key = target_name.lower()
        shape_key = f'gamma_shape_{dict_key}'
        
        # SECURITY CHECK
        if shape_key not in dna_players[position]:
            ax.set_title(f"Position: {position} (Metric not applicable)", fontsize=14)
            continue
            
        # Filter for the specific position AND strictly positive values
        positive_data = df[(df['position'] == position) & (df[column_name] > 0)][column_name]
        
        if len(positive_data) == 0:
            ax.set_title(f"Position: {position} (No positive data)", fontsize=14)
            continue
            
        shape = dna_players[position][shape_key]
        scale = dna_players[position][f'gamma_scale_{dict_key}']
        
        # Histogram of only the positive values
        sns.histplot(
            positive_data, 
            stat='density', 
            color='goldenrod', 
            bins=20,
            label=f'Real Data ({position} > 0)',
            ax=ax
        )
        
        # Gamma curve generation
        x = np.linspace(0.001, positive_data.max(), 100)
        y = stats.gamma.pdf(x, shape, loc=0, scale=scale)
        
        ax.plot(x, y, color='crimson', linewidth=3, label=f'Gamma Fit')
        
        # Plot formatting for the individual subplot
        ax.set_title(f"Position: {position}", fontsize=14)
        
        # Use the formatted display_name for the X-axis label
        ax.set_xlabel(f"{display_name} Per Game")
        
        ax.set_ylabel("Probability Density")
        ax.legend()
    
    plt.tight_layout()


# Model validation plot for Beta fit on penalty conversion rates
def beta_plt(df: pd.DataFrame, dna_players: dict) -> None:
    """
    Generates a 2x2 subplot grid to validate the Beta fit for penalty conversion rates.
    It only considers players who have taken at least one penalty.

    Args:
        df (pd.DataFrame): The DataFrame containing active player data;
        dna_players (dict): A dictionary containing the fitted Beta parameters.
    """    
    positions = ['P', 'D', 'C', 'A']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Validation Beta Fit for Penalty Conversion Rate", fontsize=18, fontweight='bold')
    sns.set_theme(style="whitegrid")
    
    axes = axes.flatten()
    
    for i, position in enumerate(positions):
        ax = axes[i]
        
        # Ensure the Beta parameters exist for this position
        if 'beta_a_pen_conversion' not in dna_players[position]:
            ax.set_title(f"Position: {position} (Metric not applicable)", fontsize=14)
            continue
            
        # Filter for actual penalty takers in this position
        pen_takers = df[(df['position'] == position) & (df['penalties_taken'] > 0)]
        
        if len(pen_takers) == 0:
            ax.set_title(f"Position: {position} (No penalty takers)", fontsize=14)
            continue
            
        # Calculate real conversion rates
        conversion_rates = pen_takers['penalties_scored'] / pen_takers['penalties_taken']
        
        # Retrieve Beta parameters
        a = dna_players[position]['beta_a_pen_conversion']
        b = dna_players[position]['beta_b_pen_conversion']
        
        # Histogram of real conversion rates (Bins from 0.0 to 1.0)
        sns.histplot(
            conversion_rates, 
            stat='density', 
            color='mediumorchid', 
            bins=10,
            binrange=(0, 1),
            label=f'Real Data ({position})',
            ax=ax
        )
        
        # Beta curve generation (X goes from 0 to 1)
        x = np.linspace(0.001, 0.999, 100)
        y = stats.beta.pdf(x, a, b)
        
        ax.plot(x, y, color='crimson', linewidth=3, label=f'Beta Fit')
        
        # Plot formatting
        ax.set_title(f"Position: {position}", fontsize=14)
        ax.set_xlabel("Conversion Rate (0.0 to 1.0)")
        ax.set_ylabel("Probability Density")
        
        # Set X limits strictly between 0 and 1 (0% to 100%)
        ax.set_xlim(-0.05, 1.05)
        ax.legend()
    
    plt.tight_layout()


