#!/bin/bash
#SBATCH -J my_experiment     # Name that will show up in squeue
#SBATCH --gres=gpu:4         # Request 4 GPU "generic resource"
#SBATCH --time=0-03:00       # Max job time is 3 hours
#SBATCH --mem=32G            # Max memory (CPU) 16GB
#SBATCH --cpus-per-task=4    # Request 4 CPU threads
#SBATCH --output=%N-%j.out   # Terminal output to file named (hostname)-(jobid).out
#SBATCH --nodelist=cs-venus-07   # if needed, set the node you want (similar to -w xyz)

# The SBATCH directives above set options similarly to command line arguments to srun
# Run this script with: sbatch my_experiment.sh
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel squ<job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_env
hostname

# Note the actual command is run through srun
srun python test.py
