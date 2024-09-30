import matplotlib.pyplot as plt
import numpy as np

# Example data for each chart
metrics = ['rule_based_agent', 'peaceful_agent', 'random_agent', 'sarsa_agent', 'ppo_agent', 'gp_agent']
# Phase 1 data
values_list = [
    [47, 0, 0, 21, 2, 2],  # !金币
    [190, 0, 1, 47.2, 4.58, 1],  # !炸箱子数量
    [400, 250, 3, 394.9, 253, 0.75],  # Data for chart 3
    [1, 0.85, 0.80, 0,0, 0] ]  # 

titles = ['Score', 'Crates Destroyed', 'Valid Action', 'Opponent Killed']

# Loop through each dataset and save them as individual plots
for i, values in enumerate(values_list):
    fig, ax = plt.subplots(figsize=(6, 5))  # Create a new figure for each plot
    
    # Dynamically determine y-axis limit based on the max value for each chart
    max_value = max(values)
    ax.set_ylim(0, max_value * 1.2)  # Set y-axis limit slightly above max

    # Plot the bars with narrower width
    bars = ax.bar(metrics, values, width=0.4, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#8172b3', '#8172b3'])
    
    # Rotate x-axis labels to prevent overlap
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=10)
    
    # Customize each subplot
    ax.set_title(titles[i], fontsize=12)  # Reduced title font size, no bold
    
    # Remove the "Values" label from y-axis but keep the ticks
    ax.set_ylabel('')  # This removes the label while keeping y-axis ticks
    
    # Add gridlines for clarity
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Save each chart as a separate PNG file
    plt.tight_layout()
    plt.savefig(f'{titles[i].replace(" ", "_").lower()}.png', dpi=300)
    
    plt.close(fig)  # Close the figure to avoid overlapping plots