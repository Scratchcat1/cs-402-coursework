set -e
#filename="trapActuallySerial"
# filename="trapSerial2"
# gcc -fopenmp -O3 -o "$filename" "${filename}.c"
srun_output=$(sbatch submit.sbatch)
echo "$srun_output"
arr=($srun_output)
echo "${arr[3]}"
sleep 5s
tail -f "slurm-${arr[3]}.out"
