import os
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from collections import Counter
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
def truncate(value, decimal_places=2):
    """
    Truncates a number to a specified number of decimal places without rounding.
    
    Args:
        value (float): The number to truncate.
        decimal_places (int): The number of decimal places to keep.
    
    Returns:
        float: The truncated number.
    """
    factor = 10.0 ** decimal_places
    return int(value * factor) / factor

def pad_scores(scores_dict):
    """
    Pads the lists of scores to the same length with NaNs.
    
    Args:
        scores_dict (dict): A dictionary with keys as seeds and values as lists of scores.
    
    Returns:
        np.array: A 2D numpy array with padded scores.
    """
    max_length = max(len(scores) for scores in scores_dict.values())
    padded_scores = []
    for scores in scores_dict.values():
        padded_scores.append(np.pad(scores, (0, max_length - len(scores)), constant_values=np.nan))
    return np.array(padded_scores)

# Function to plot scores from the dictionary without the median
def plot_scores_without_median(scores_dict, save_dir='src/plots/solo_runs', x_tick_interval=50):
    """
    Plot the scores over episodes for each seed and save the plot.
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
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Scores during PPO Training')
    plt.grid(True)

    # Set x-axis limits and ticks
    plt.xlim(0, max(len(scores) for scores in scores_dict.values()))
    plt.xticks(np.arange(0, max(len(scores) for scores in scores_dict.values()) + 1, step=x_tick_interval))

    # Remove scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    plt.legend()  # Show the legend with seed info

    # Ensure the directory exists
    ensure_directory_exists(save_dir)

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'scores_baselines.png'))
    plt.close()

def plot_mean_and_ci(scores_dict, save_dir='src/plots/solo_runs', x_tick_interval=50):
    """
    Plot the mean scores with confidence intervals over episodes for multiple seeds.
    """
    if not scores_dict:
        print("No scores to plot.")
        return

    plt.figure(figsize=(10, 5))

    # Padding the scores to ensure equal length
    all_scores = pad_scores(scores_dict)

    # Compute mean and standard deviation, ignoring NaNs
    mean_scores = np.nanmean(all_scores, axis=0)
    std_scores = np.nanstd(all_scores, axis=0)

    # Plot mean scores
    plt.plot(mean_scores, label='Mean Score', color='blue')

    # Fill the standard deviation area
    plt.fill_between(range(len(mean_scores)), mean_scores - std_scores, mean_scores + std_scores, 
                     color='blue', alpha=0.2, label='Std Dev')

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Mean Scores during PPO Training with Std Dev')
    plt.grid(True)

    # Set x-axis limits and ticks
    plt.xlim(0, len(all_scores))
    plt.xticks(np.arange(0, len(mean_scores) + 1, step=x_tick_interval))

    # Remove scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    plt.legend()  # Show the legend with mean and std dev

    # Ensure the directory exists
    ensure_directory_exists(save_dir)

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'mean_scores_with_std_dev.png'))

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

def plot_value_frequencies_heatmap(values, save_dir='src/plots/solo_runs', decimal_places=2):
    """
    Plot a heatmap of the frequency of each truncated value.
    """
    if not values:
        print("No values to plot.")
        return

    # Truncate values to the desired number of decimal places
    truncated_values = [truncate(v, decimal_places) for v in values]
    
    # Count the frequency of each truncated value
    value_counts = Counter(truncated_values)
    
    # Convert to two lists for the heatmap
    unique_values = list(value_counts.keys())
    frequencies = list(value_counts.values())
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([frequencies]), annot=True, fmt='d', cmap='viridis', xticklabels=unique_values)
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Frequency of Truncated Values')
    plt.yticks([])  # Hide the y-axis ticks
    
    # Ensure the directory exists
    ensure_directory_exists(save_dir)
    
    # Save the heatmap
    plt.savefig(os.path.join(save_dir, 'value_frequencies_heatmap.png'))
    plt.close()


def main():
    # Path to your JSON file with scores
    json_file_path = "logdir/crafter_reward-ppo/1/scores.json"

    # Load the JSON data
    data = load_json_file(json_file_path)
    keys_to_process = ["42", "111", "101", "292", "300"]  # Add all your desired keys here
    
    # Extract and combine values from all specified keys
    combined_values = []
    for key in keys_to_process:
        combined_values.extend(data.get(key, []))
    # Plot the scores without the median with a custom x-axis tick interval
    plot_scores_without_median(data, x_tick_interval=200)  # Adjust x_tick_interval as needed
    plot_mean_and_ci(data, x_tick_interval=200)
    plot_value_frequencies_heatmap(combined_values)

if __name__ == "__main__":
    main()