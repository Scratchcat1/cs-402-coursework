set -e
export OMP_NUM_THREADS=6
make clean
make -j 8
cd build
./deqn ../test/square.in

valgrind --tool=callgrind ./deqn ../test/big_square.in
