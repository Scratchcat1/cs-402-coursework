set -e
export OMP_NUM_THREADS=4
make clean
make -j 8
cd build
./deqn ../test/square.in
./deqn ../test/big_square.in
valgrind --tool=callgrind ./deqn ../test/big_square.in
