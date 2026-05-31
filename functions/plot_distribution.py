import math             # For mathematical functions, specifically for calculating grid dimensions
import pandas as pd     # For data manipulation
import seaborn as sns   # For data visualization
import matplotlib.pyplot as plt     # For data visualization

def plot_distribution(df: pd.DataFrame, target_column: str, category_column: str = None, title: str = "Distribution Plot", num_bins: int = 39, max_x: int = 38) -> None:
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
                kde=True,
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