#!/bin/bash
#SBATCH --job-name=cfd
#SBATCH --partition=cs402
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=00:10:00
. /etc/profile.d/modules.sh
module purge
module load cs402-mpi
# time mpirun ./a.out
time ./karman --infile initial.bin -o karman.bin -t 25
gprof ./karman
./bin2ppm < karman.bin > karman.ppm
./diffbin karman.vanilla.bin karman.bin