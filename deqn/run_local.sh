set -e
# export OMP_NUM_THREADS=1
make clean
make -j 8
cd build
./deqn ../test/square.in
#./deqn ../test/big_square.in
#./deqn ../test/long_running_square.in
#./deqn ../test/boundary_square.in
#./deqn ../test/mega_ultra_square.in
# valgrind --tool=callgrind ./deqn ../test/big_square.in
