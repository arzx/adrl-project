#!/bin/bash -l
#SBATCH --job-name=test_serial
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:20:00
#SBATCH --constraint=[skylake|haswell]
#SBATCH --mail-user=artur.ganzha@stud.uni-hannover.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output test_serial-job_%j.out
#SBATCH --error test_serial-job_%j.err
# Change to my work dir
# SLURM_SUBMIT_DIR is an environment variable that automatically gets
# assigned the directory from which you did submit the job. A batch job
# is like a new login, so you'll initially be in your HOME directory.
# So it's usually a good idea to first change into the directory you did # submit your job from.
cd $SLURM_SUBMIT_DIR
# Load the modules you need, see corresponding page in the cluster documentation
module load python/3.12.4
source activate adrl-project-luis 
# Start my serial app
# srun is needed here only to create an entry in the accounting system,
python baselines.py 
# but you could also start your app without it here, since it's only serial. srun ./my_serial_app
srun ./baselines.py