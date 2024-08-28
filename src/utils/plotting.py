import os
import matplotlib.pyplot as plt
import json
import numpy as np

# Ensure directory exists
def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

def load_json_file(file_path):
    """
    Loads a JSON file and returns the data as a Python dictionary.
    
    Args:
        file_path (str): The path to the JSON file.
    
    Returns:
        dict: The data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to plot scores from the dictionary and the median
def plot_scores(scores_dict, save_dir='src/plots/baselines', x_tick_interval=50):
    """
    Plot the scores over episodes for each seed and save the plot.
    Also plot the median of the scores across seeds.
    """
    if not scores_dict:
        print("No scores to plot.")
        return

    plt.figure(figsize=(10, 5))

    # Set a color map
    colors = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors

    # Plot individual seed scores
    for i, (seed, scores) in enumerate(scores_dict.items()):
        plt.plot(scores, label=f'Seed {seed}', color=colors(i % 10))

    # Compute and plot the median scores across seeds
    all_scores = np.array(list(scores_dict.values()))
    median_scores = np.median(all_scores, axis=0)
    plt.plot(median_scores, label='Median', color='black', linestyle='--', linewidth=2)

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Scores during PPO Training with Median')
    plt.grid(True)

    # Set x-axis limits and ticks
    plt.xlim(0, len(median_scores))
    plt.xticks(np.arange(0, len(median_scores) + 1, step=x_tick_interval))

    # Remove scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    plt.legend()  # Show the legend with seed info

    # Ensure the directory exists
    ensure_directory_exists(save_dir)

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'scores_baselines_with_median.png'))
    plt.close()

def plot_mean_and_ci(scores_dict, save_dir='src/plots/baselines', x_tick_interval=50):
    """
    Plot the mean scores with confidence intervals over episodes for multiple seeds.
    """
    if not scores_dict:
        print("No scores to plot.")
        return

    plt.figure(figsize=(10, 5))

    all_scores = np.array(list(scores_dict.values()))
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)

    plt.plot(mean_scores, label='Mean Score', color='blue')
    plt.fill_between(range(len(mean_scores)), mean_scores - std_scores, mean_scores + std_scores, color='blue', alpha=0.2, label='Std Dev')

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Mean Scores during PPO Training with Std Dev')
    plt.grid(True)

    # Set x-axis limits and ticks
    plt.xlim(0, len(mean_scores))
    plt.xticks(np.arange(0, len(mean_scores) + 1, step=x_tick_interval))

    # Remove scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    plt.legend()  # Show the legend with mean and std dev

    # Ensure the directory exists
    ensure_directory_exists(save_dir)

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'median.png'))

    plt.close()

def save_json_file(data, file_path):
    """
    Saves a Python dictionary or list to a JSON file.
    
    Args:
        data (dict or list): The data to save.
        file_path (str): The path to the JSON file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        print(f"Data successfully saved to {file_path}")

def main():
    # Path to your JSON file with scores
    json_file_path = 'logdir/crafter_reward-ppo/0/scores.json'
    
    # Load the JSON data
    data = load_json_file(json_file_path)
    
    # Plot the scores and the median with a custom x-axis tick interval
    plot_scores(data, x_tick_interval=50)  # Adjust x_tick_interval as needed
    plot_mean_and_ci(data, x_tick_interval=100)
if __name__ == "__main__":
    main()