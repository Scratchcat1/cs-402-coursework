#!/bin/bash
#SBATCH --job-name=hello-world-omp # Job name for tracking
#SBATCH --partition=cs402 # Partition you wish to use
#SBATCH --nodes=1 # Number of compute nodes used by your job
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12 # Number of CPU cores used on each node
#SBATCH --time=00:05:00
module load cs402-mpi
srun bash ./run_local.sh
#srun ./trapActuallySerial
