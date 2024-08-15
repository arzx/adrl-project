#!/bin/bash -l
#SBATCH --job-name=test_serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --partition=ai
#SBATCH --mail-user=artur.ganzha@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test_serial-job_%j.out
#SBATCH --error=test_serial-job_%j.err

# Change to the directory where your script is located
cd /bigwork/nhwpgana/adrl-project

# Load the correct Python module
module spider python  # Check for available versions
module load python/3.12.4  # Load the correct Python version

# If conda is installed and available
export PATH=/path/to/conda/bin:$PATH  # Add Conda to the PATH if needed
source activate adrl-project-luis || conda activate adrl-project-luis  # Activate your conda environment

# Run the Python script
python src/agents/baselines.py --steps 100