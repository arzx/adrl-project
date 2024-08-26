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

# Function to plot scores from the dictionary
def plot_scores(scores_dict, save_dir='src/plots/baselines'):
    """
    Plot the scores over episodes for each seed and save the plot.
    """
    if not scores_dict:
        print("No scores to plot.")
        return

    plt.figure(figsize=(10, 5))

    # Set a color map
    colors = plt.get_cmap('tab10')  # 'tab10' has 10 distinct colors

    for i, (seed, scores) in enumerate(scores_dict.items()):
        plt.plot(scores, label=f'Seed {seed}', color=colors(i % 10))

    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.title('Scores during PPO Training')
    plt.grid(True)

    # Remove scientific notation on the y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useOffset=False))

    plt.legend()  # Show the legend with seed info

    # Ensure the directory exists
    ensure_directory_exists(save_dir)

    # Save the plot
    plt.savefig(os.path.join(save_dir, 'scores_baselines.png'))
    plt.close()

# Save a JSON file (if needed)
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
    # Example data (replace this with your actual JSON data loading)
    json_file_path = 'logdir/crafter_reward-ppo/0/scores.json'
    
    # Load the JSON data
    data = load_json_file(json_file_path)
    # Plot the scores
    plot_scores(data)

if __name__ == "__main__":
    main()