#!/bin/bash -l
#SBATCH --job-name=test_serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --partition=ai         # Explicitly set the partition to 'ainlp'
#SBATCH --mail-user=artur.ganzha@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test_serial-job_%j.out
#SBATCH --error=test_serial-job_%j.err

# Change to the directory from which the job was submitted
cd /bigwork/nhwpgana/adrl-project

# Load the necessary module
module load python/3.12.4

# Activate the conda environment (use the correct syntax for your system)
source activate adrl-project-luis || conda activate adrl-project-luis

# Run the Python script without srun
python baselines.py