#!/bin/bash
#SBATCH --job-name=cfd
#SBATCH --partition=cs402
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=2
##SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=12
##SBATCH --ntasks-per-socket=1
#SBATCH --time=00:10:00
. /etc/profile.d/modules.sh
module purge
module load cs402-mpi
# time mpirun ./a.out
lscpu
cat /proc/cpuinfo
export OMP_NUM_THREADS=12
time mpirun -report-bindings echo
time mpirun -n 1 --use-hwthread-cpus --bind-to socket -display-map -display-allocation ./karman-par -v 3 -x 1000 -y 800 --infile initial-big.bin -o karman-big.bin -t 0.02
#time ./karman -x 100 -y 100 --infile initial-big.bin -o karman-big.bin -t 25
#gprof ./karman
./bin2ppm < karman.bin > karman.ppm
./diffbin karman.vanilla.bin karman.bin

#mpirun -n 2 ./pingpong
