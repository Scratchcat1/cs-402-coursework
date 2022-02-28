#!/bin/bash
#SBATCH --job-name=cfd
#SBATCH --partition=cs402
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
. /etc/profile.d/modules.sh
module purge
module load cs402-mpi
# time mpirun ./a.out
cat /proc/cpuinfo
export OMP_NUM_THREADS=1
time mpirun -npersocket 1 --bind-to socket ./karman-par -x 1000 -y 800 --infile initial-big.bin -o karman-big.bin -t 25
#time ./karman -x 100 -y 100 --infile initial-big.bin -o karman-big.bin -t 25
#gprof ./karman
./bin2ppm < karman.bin > karman.ppm
./diffbin karman.vanilla.bin karman.bin

# mpirun -n 2 ./pingpong
