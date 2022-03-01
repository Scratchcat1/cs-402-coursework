#!/usr/bin/python3
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
dimensions = [(1000, 200), (2000, 400)]
omp_num_threads_tested = [1, 2, 3, 4]

def get_time_from_timing_line(line):
    string_time = line.split(" ")[3]
    return float(string_time)

class CFDRunner:
    def __init__(self, id):
        self.x = 660
        self.y = 120
        self.t = 0.2
        self.sbatch_nodes = 1
        self.sbatch_tasks = 0.5
        self.sbatch_time = "00:10:00"
        self.omp_threads = 6
        self.in_file = os.path.join("test", f"initial-{id}.bin")
        self.out_file = os.path.join("test", f"completed-{id}.bin")
        self.sbatch_file = os.path.join("test", f"submit-{id}.sbatch")
        self.single_thread = False

    def run(self):
        process_output = subprocess.run(["sbatch", self.sbatch_file], stdout=subprocess.PIPE)
        output_lines = process_output.stdout.decode().split("\n")
        self.sbatch_id = output_lines[0].split(" ")[3]

    def is_still_running(self):
        process_output = subprocess.run(["squeue"], stdout=subprocess.PIPE)
        output_lines = process_output.stdout.decode().split("\n")
        return any([self.sbatch_id in line for line in output_lines])

    def parse_output(self):
        with open(f"slurm-{self.sbatch_id}.out", "r") as fh:
            lines = fh.readlines()

        i = 0
        while "I am process" not in lines[i]:
            i += 1

        shape_output = lines[i]

        timing_results = []

        current_time = None
        timestep_time_taken = None
        compute_velocity_time_taken = None
        rhs_time_taken = None
        possion_time_taken = None
        update_velocity_time_taken = None
        boundary_time_taken = None
        for line in lines[i:]:
            if "--- Timestep" in line:
                if current_time is not None:
                    timing_results.append([
                        current_time,
                        timestep_time_taken,
                        compute_velocity_time_taken,
                        rhs_time_taken,
                        possion_time_taken,
                        update_velocity_time_taken,
                        boundary_time_taken
                        ])

                current_time = float(line.split(" ")[3])

            elif "timestep_time_taken" in line:
                timestep_time_taken = float(line.split(" ")[1])

            elif "compute_velocity_time_taken" in line:
                compute_velocity_time_taken = float(line.split(" ")[1])

            elif "rhs_time_taken" in line:
                rhs_time_taken = float(line.split(" ")[1])

            elif "possion_time_taken" in line:
                possion_time_taken = float(line.split(" ")[1])

            elif "update_velocity_time_taken" in line:
                update_velocity_time_taken = float(line.split(" ")[1])

            elif "boundary_time_taken" in line:
                boundary_time_taken = float(line.split(" ")[1])
        
        df = pd.DataFrame(timing_results, columns=("Timestep", "timestep_time_taken", "compute_velocity_time_taken", "rhs_time_taken", "possion_time_taken", "update_velocity_time_taken", "boundary_time_taken"))
        return df


    
    def save_sbatch(self):
        command = f"time mpirun -npersocket 1 --bind-to socket ./karman-par -x {self.x} -y {self.y} --infile {self.in_file} -o {self.out_file} -t {self.t}\n",
        if self.single_thread:
            command = f"time ./karman -x {self.x} -y {self.y} --infile {self.in_file} -o {self.out_file} -t {self.t}\n",
        with open(self.sbatch_file, "w") as fh:
            fh.writelines([
                "#!/bin/bash\n",
                "#SBATCH --job-name=cfd\n",
                "#SBATCH --partition=desktop-batch\n",
                "#SBATCH --nice=9000\n",
                "#SBATCH --ntasks-per-socket=1\n",
                f"#SBATCH --nodes={self.sbatch_nodes}\n",
                f"#SBATCH --ntasks={self.sbatch_tasks}\n",
                f"#SBATCH --cpus-per-task={min(6, self.omp_threads)}\n"
                f"#SBATCH --time={self.sbatch_time}\n",
                ". /etc/profile.d/modules.sh\n",
                "module purge\n",
                "module load cs402-mpi\n",
                f"export OMP_NUM_THREADS={self.omp_threads}\n",
                command,
                "#gprof ./karman\n",
                "./bin2ppm < karman.bin > karman.ppm\n",
                "./diffbin karman.vanilla.bin karman.bin\n",
            ])

def collect_data():
    all_df = pd.DataFrame({
        "x": pd.Series(dtype='int32'),
        "y": pd.Series(dtype='int32'),
        "sbatch_nodes": pd.Series(dtype='int32'),
        "sbatch_tasks": pd.Series(dtype='int32'),
        "omp_threads": pd.Series(dtype='int32'),
        "current_time": pd.Series(dtype='float'),
        "timestep_time_taken": pd.Series(dtype='float'),
        "compute_velocity_time_taken": pd.Series(dtype='float'),
        "rhs_time_taken": pd.Series(dtype='float'),
        "possion_time_taken": pd.Series(dtype='float'),
        "update_velocity_time_taken": pd.Series(dtype='float'),
        "boundary_time_taken": pd.Series(dtype='float'),
        })
    id = 0
    runners = []
    for x, y in dimensions:
        id += 1
        st_runner = CFDRunner(id)
        st_runner.single_thread = True
        st_runner.x = x
        st_runner.y = y
        st_runner.sbatch_tasks = sbatch_tasks
        st_runner.omp_threads = 0
        st_runner.save_sbatch()
        runners.append(st_runner)

        for sbatch_nodes in [1, 2, 3, 4]:
            for omp_num_threads in omp_num_threads_tested:
                csv_path = os.path.join("timing_data", f"{x}-{y}-{sbatch_nodes}-{omp_num_threads}.csv")
                if os.path.exists(csv_path):
                    continue
                id += 1
                sbatch_tasks = int(sbatch_nodes * np.ceil(omp_num_threads / 6))
                print(sbatch_tasks, sbatch_nodes, omp_num_threads)
                cfd_runner = CFDRunner(id)
                cfd_runner.x = x
                cfd_runner.y = y
                cfd_runner.sbatch_nodes = sbatch_nodes
                cfd_runner.sbatch_tasks = sbatch_tasks
                cfd_runner.omp_threads = omp_num_threads
                cfd_runner.save_sbatch()
                runners.append(cfd_runner)
    max_running = 1
    to_be_run = runners[::-1]
    while len(to_be_run) > 0:
        running_list = []
        while len(running_list) < max_running and len(to_be_run) > 0:
            runner = to_be_run.pop()
            print(runner.x, runner.y, runner.sbatch_nodes, runner.omp_threads)
            runner.run()
            running_list.append(runner)
        for cfd_runner in running_list:
            time.sleep(5)
            while cfd_runner.is_still_running():
                time.sleep(1)
            csv_path = os.path.join("timing_data", f"{cfd_runner.x}-{cfd_runner.y}-{cfd_runner.sbatch_nodes}-{cfd_runner.omp_threads}.csv")
            df = cfd_runner.parse_output()
            df["x"] = cfd_runner.x
            df["y"] = cfd_runner.y
            df["sbatch_nodes"] = cfd_runner.sbatch_nodes
            df["sbatch_tasks"] = cfd_runner.sbatch_tasks
            df["omp_threads"] = cfd_runner.omp_threads
            df.to_csv(csv_path)
    #         all_df = pd.concat([all_df, df])
    #         print(all_df)
    # all_df.to_csv("timing.csv")

def plot_graphs():
    filenames = glob.glob("timing_data/*.csv")
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)
    all_df = pd.concat(dfs, axis=0, ignore_index=True)
    # print(all_df)
    all_df["loop_time_taken"] = all_df[["timestep_time_taken", "compute_velocity_time_taken", "rhs_time_taken", "possion_time_taken", "update_velocity_time_taken", "boundary_time_taken"]].sum(axis=1)
    plot_time_against_thread_count(all_df)
    plot_speed_up_against_thread_count(all_df)

def plot_time_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c"]
    line_styles = ["-", "--", "-."]
    for (x, y), line_style in zip(dimensions, line_styles):
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for sbatch_nodes, colour in zip(dim_df["sbatch_nodes"].unique(), colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            plt.plot(node_df["omp_threads"], node_df["loop_time_taken"], colour + line_style, label=f"{sbatch_nodes} - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.savefig("plots/time_against_thread_count.png", dpi=600)
    plt.clf()

def plot_speed_up_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c"]
    line_styles = ["-", "--", "-."]
    for (x, y), line_style in zip(dimensions, line_styles):
        df_st_dim = df_st[df_st["x"] == x]
        df_st_dim = df_st_dim[df_st_dim["y"] == y]
        st_time_taken = df_st_dim["loop_time_taken"].iloc[0]
        print(st_time_taken)
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for sbatch_nodes, colour in zip(dim_df["sbatch_nodes"].unique(), colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            plt.plot(node_df["omp_threads"], st_time_taken / node_df["loop_time_taken"], colour + line_style, label=f"{sbatch_nodes} - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.savefig("plots/speed_up_against_thread_count.png", dpi=600)
    plt.clf()


if __name__ == "__main__":
    # subprocess.run(["bash", "./clean_build.sh"])
    # collect_data()
    plot_graphs()
