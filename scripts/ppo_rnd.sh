#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J baselines
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o slurm_outputs/slurm-%j.out
#SBATCH --partition=tnt,ai
#SBATCH --time=8:00:00
#SBATCH --mail-user=artur.ganzha@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=test_serial-job_%j.out
#SBATCH --error=test_serial-job_%j.err

cd adrl-project/slurm_outputs/

module load Miniconda3

source /home/nhwpgana/.bashrc
conda activate /bigwork/nhwpgana/.conda/envs/adrl-project-luis
export WANDB_MODE=offline
export PYTHONPATH=$(pwd)/src:$PYTHONPATH


# Run the Python script
python /bigwork/nhwpgana/adrl-project/src/agents/ppo_rnd_pytorch.py --steps 1000000