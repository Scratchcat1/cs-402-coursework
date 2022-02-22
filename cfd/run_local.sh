set -e
make clean all

echo "\nRunning karman"
mpirun ./karman --infile initial.bin -o karman.bin -t 25 && ./bin2ppm < karman.bin > karman.ppm && ./diffbin karman.vanilla.bin karman.bin