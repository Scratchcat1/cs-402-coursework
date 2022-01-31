set -e
# export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Running on file: $1"
#make clean
#make -j 8
cd build
./deqn "$1"
