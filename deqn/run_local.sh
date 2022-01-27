set -e
export OMP_NUM_THREADS=4
make clean
make
cd build
./deqn ../test/square.in

./deqn ../test/big_square.in
