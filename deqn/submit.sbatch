#!/bin/bash
#SBATCH --job-name=cs402-cswk-1 # Job name for tracking
#SBATCH --partition=cpu-batch #cs402 # Partition you wish to use
#SBATCH --nodes=1 # Number of compute nodes used by your job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12 # Number of CPU cores used on each node
#SBATCH --time=00:10:00
module load cs402-mpi
srun python3 analysis.py
#srun ./trapActuallySerial
