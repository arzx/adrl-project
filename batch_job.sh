#!/bin/bash -l
#SBATCH --job-name=adrl-baselines
#SBATCH --gres=gpu:a100:2  # Request 2 A100 GPUs
#SBATCH --mem=64G  # Request 64GB of memory (adjust as needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # Adjust the number of CPUs per task if needed
#SBATCH --time=08:00:00  # Adjust time based on expected runtime
#SBATCH --partition=ai  # Partition for AI workloads
#SBATCH --mail-user=artur.ganzha@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test_serial-job_%j.out
#SBATCH --error=test_serial-job_%j.err

# Change to the directory where your script is located
cd /bigwork/nhwpgana/adrl-project

# Load the correct Python module
module spider python  # Check for available versions
module load python/3.12.4  # Load the correct Python version

# Activate your conda environment
source activate adrl-project-luis || conda activate adrl-project-luis

# Run the Python script
python src/agents/baselines.py --steps 1000000