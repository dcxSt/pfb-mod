#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:29:00
#SBATCH --job-name upchan_stats_debug
#SBATCH --output=upchan_stats_debug_%j.txt
#SBATCH --mail-user=stephen.fay@mail.mcgill.ca
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

module load python/3.8

srun bash -c 'python3.8 snr_analysis_corr.py $((10000 + SLURM_PROCID)) 1'



# To use debug node:
# #SBATCH --partition=debug

## use srun to launch tasks
#TASK_FILE="tasks.txt"
#rm -f $TASK_FILE
#touch $TASK_FILE
#
#for i in {1..64}; do
#    echo "python3.8 snr_analysis_corr.py $((1100 + i)) 1" >> $TASK_FILE
#done
#
## Run tasks using srun in parallel
#srun --multi-prog $TASK_FILE


