import matplotlib.pyplot as plt
import numpy as np

# Example data for each chart
metrics = ['coin_collector_agent', 'peaceful_agent', 'random_agent', 'sarsa_agent', 'ppo_agent', 'gp_agent']
# phase 1
values_list = [
    [50, 30, 1, 42.83, 18.72, 13],  # ！金币
    [0.78, 0.73, 0.68, 0.74, 0.81,0],  # Score
    [400, 270, 3, 400, 253, 7],  # !Valid Action
    [0.88, 0.85, 0.80, 0.79, 0.84,0]   # Crates Destroyed
]
# phase 2
# values_list = [
#     [39, 12.5, 1, 15.09, 10.97, 2],  # ！金币
#     [0.78, 0.73, 0.68, 0.74, 0.81],  # Data for chart 2
#     [400, 250, 3, 394.9, 255, 7],  # ！有效步数
#     [0.88, 0.85, 0.80, 0.79, 0.84]   # Data for chart 4
# ]

# phase 3
# values_list = [
#     [47, 0, 0, 21, 2, 2],  # !金币
#     [190, 0, 1, 47.2, 4.58, ],  # !炸箱子数量
#     [400, 250, 3, 394.9, 253, 0.75],  # Data for chart 3
#     [1, 0.85, 0.80, 0, 0]   # 
# phase 4
# values_list = [
#     [52, 2, 2, 19, 5, 2],  # !分数
#     [190, 0, 1, 4.7, 2.58, 0],  # 炸箱子数量
#     [400, 250, 3, 394.9, 253, 0.75],  # Data for chart 3
#     [1.6, 0, 0, 0.2, 0,0]   # !杀敌数量, 第一个改为rule_based agent
# ]

titles = ['Coins Collected', 'Score', 
          'Valid Action', 'Crates Destroyed']

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Flatten the 2x2 array of axes to easily iterate
axs = axs.flatten()

# Loop through each axis and plot a bar chart
for i, ax in enumerate(axs):
    bars = ax.bar(metrics, values_list[i], color=['#4c72b0', '#55a868', '#c44e52', '#8172b3', '#8172b3'])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset the text above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Customize each subplot
    ax.set_ylim(0, 1)  # Set y-axis limit
    ax.set_title(titles[i], fontsize=14, fontweight='bold')
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

# Adjust the layout so the subplots don't overlap
plt.tight_layout()

# Save the figure as a high-resolution image for use in a paper
plt.savefig('performance_metrics_4_charts.png', dpi=300)

# Show the plot
plt.show()