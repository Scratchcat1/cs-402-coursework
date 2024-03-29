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

# Set up a bunch of settings to test, more than will be plotted to ensure that I can change things around to plot different values
dimensions = [(660, 120), (1000, 200), (2000, 400), (4000, 800), (8000, 1600)]
sim_times = [0.5, 0.2, 0.05, 0.01, 0.004]
omp_num_threads_tested = [1, 2, 3, 4, 5, 6]
sbatch_nodes_tested = [1, 2, 3, 4, 8]

# List of items to actually plot as different lines
omp_num_threads_plot = [1, 2, 4, 6]
sbatch_nodes_plot = [1, 2, 4, 8]
dimensions_plot = [(660, 120), (2000, 400), (8000, 1600)]

# Extract timing data from line
def get_time_from_timing_line(line):
    string_time = line.split(" ")[3]
    return float(string_time)

class CFDRunner:
    """
    Class to handle running a configuration via slurm and process the output
    """
    def __init__(self, id):
        """
        Set the default parameters
        Takes an id to keep config files separate
        """
        self.x = 660
        self.y = 120
        self.t = 0.2
        self.sbatch_nodes = 1
        self.sbatch_tasks = 0.5
        self.sbatch_time = "00:07:00"
        self.omp_threads = 6
        self.in_file = os.path.join("test", f"initial-{id}.bin")
        self.out_file = os.path.join("test", f"completed-{id}.bin")
        self.sbatch_file = os.path.join("test", f"submit-{id}.sbatch")
        self.single_thread = False

    def run(self):
        """
        Run the slurm batch file and extract the slurm job id to read the file later
        """
        process_output = subprocess.run(["sbatch", self.sbatch_file], stdout=subprocess.PIPE)
        output_lines = process_output.stdout.decode().split("\n")
        self.sbatch_id = output_lines[0].split(" ")[3]

    def is_still_running(self):
        """
        Check if the job still appears in the queue -> probably still running
        """
        process_output = subprocess.run(["squeue"], stdout=subprocess.PIPE)
        output_lines = process_output.stdout.decode().split("\n")
        return any([self.sbatch_id in line for line in output_lines])

    def parse_output(self):
        """
        Parse the output into a dataframe of timing data
        """
        with open(f"slurm-{self.sbatch_id}.out", "r") as fh:
            lines = fh.readlines()

        i = 0
#        while i < len(lines) and "I am process" not in lines[i]:
#            i += 1

#        shape_output = lines[i]

        timing_results = []

        # Basically go line by line and extract the timing data
        # If a timestep label is seen it knows a new set of measurements is starting
        # Add the current of the data to the dataframe
        # Note: Uses this weird method because it wasn't known which order the measurements would be output in

        current_time = None
        timestep_time_taken = None
        compute_velocity_time_taken = None
        rhs_time_taken = None
        possion_time_taken = None
        update_velocity_time_taken = None
        boundary_time_taken = None
        sync_time_taken = None
        possion_p_loop_time_taken = None
        possion_res_loop_time_taken = None
        for line in lines[i:]:
            try:
                if "--- Timestep" in line:
                    if current_time is not None:
                        timing_results.append([
                            current_time,
                            timestep_time_taken,
                            compute_velocity_time_taken,
                            rhs_time_taken,
                            possion_time_taken,
                            update_velocity_time_taken,
                            boundary_time_taken,
                            sync_time_taken,
                            possion_p_loop_time_taken,
                            possion_res_loop_time_taken,
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
                
                elif "sync_time_taken" in line:
                    sync_time_taken = float(line.split(" ")[1])

                elif "possion_p_loop_time_taken" in line:
                    possion_p_loop_time_taken = float(line.split(" ")[1])

                elif "possion_res_loop_time_taken" in line:
                    possion_res_loop_time_taken = float(line.split(" ")[1])
            except Exception as e:
                print("Exception", e)
        
        # Label the dataframe columns and return
        df = pd.DataFrame(timing_results, columns=("Timestep", "timestep_time_taken", "compute_velocity_time_taken", "rhs_time_taken", "possion_time_taken", "update_velocity_time_taken", "boundary_time_taken", "sync_time_taken", "possion_p_loop_time_taken", "possion_res_loop_time_taken"))
        return df


    
    def save_sbatch(self):
        """
        Export the configuration as a file to be run by sbatch
        Bind to socket to avoid performing openmp across two sockets to avoid memory latency
        """
        # Default to using the parallel implementation
        command = f"time mpirun -n {self.sbatch_nodes} -npernode 1 --bind-to socket ./karman-par -x {self.x} -y {self.y} --infile {self.in_file} -o {self.out_file} -t {self.t}\n"
        omp_line = f"export OMP_NUM_THREADS={self.omp_threads}\n"

        # If singlethread use the other executable
        if self.single_thread:
            command = f"time ./karman -x {self.x} -y {self.y} --infile {self.in_file} -o {self.out_file} -t {self.t}\n"
            omp_line = "\n"

        # Write out the file
        with open(self.sbatch_file, "w") as fh:
            fh.writelines([
                "#!/bin/bash\n",
                "#SBATCH --job-name=cfd-graphs\n",
                "#SBATCH --partition=cs402\n",
                "#SBATCH --nice=9000\n",
                "#SBATCH --ntasks-per-socket=1\n",  # avoid going from socket to socket with openmp
                f"#SBATCH --nodes={self.sbatch_nodes}\n",
                f"#SBATCH --ntasks-per-node=1\n",
                f"#SBATCH --cpus-per-task=12\n"     # required for 6x scaling running on slurm scaled correctly with openmp up to 6 threads but after that failed to improve. I think it is only allocating one socket.
                f"#SBATCH --time={self.sbatch_time}\n",
                ". /etc/profile.d/modules.sh\n",
                "module purge\n",
                "module load cs402-mpi\n",
                omp_line,
                command,
                "#gprof ./karman\n",
                "./bin2ppm < karman.bin > karman.ppm\n",
                "./diffbin karman.vanilla.bin karman.bin\n",
            ])

def collect_data():
    """
    Run all configurations
    """
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
        "sync_time_taken": pd.Series(dtype='float'),
        "possion_p_loop_time_taken": pd.Series(dtype='float'),
        "possion_res_loop_time_taken": pd.Series(dtype='float'),
        })
    id = 0
    # Create a list of all programs to be run
    runners = []
    for (x, y), t in zip(dimensions, sim_times):
        # Run the single threaded version if it doesn't exist
        id += 1
        csv_path = os.path.join("timing_data", f"{x}-{y}-1-0.csv")
        print(csv_path)
        if not os.path.exists(csv_path):
            st_runner = CFDRunner(id)
            st_runner.single_thread = True
            st_runner.x = x
            st_runner.y = y
            st_runner.t = t
            st_runner.sbatch_tasks = 1
            st_runner.omp_threads = 0
            st_runner.save_sbatch()
            runners.append(st_runner)

        # for each number of nodes
        for sbatch_nodes in sbatch_nodes_tested:
            # for each number of openmp threads
            for omp_num_threads in omp_num_threads_tested:
                # Add only if the output data doesn't already exist
                csv_path = os.path.join("timing_data", f"{x}-{y}-{sbatch_nodes}-{omp_num_threads}.csv")
                if os.path.exists(csv_path):
                    continue
                id += 1
                sbatch_tasks = sbatch_nodes * 12 #int(sbatch_nodes * np.ceil(omp_num_threads / 6))
                print(sbatch_tasks, sbatch_nodes, omp_num_threads)
                cfd_runner = CFDRunner(id)
                cfd_runner.x = x
                cfd_runner.y = y
                cfd_runner.t = t
                cfd_runner.sbatch_nodes = sbatch_nodes
                cfd_runner.sbatch_tasks = sbatch_tasks
                cfd_runner.omp_threads = omp_num_threads
                cfd_runner.save_sbatch()
                runners.append(cfd_runner)

    # At most run this many slurm jobs in parallel
    # Speeds things up when generating data when no one else is using the cs402 partition
    max_running = 1
    # List of configs to be run
    to_be_run = runners[::-1]
    while len(to_be_run) > 0:
        running_list = []
        # Fill up the current list of running  jobs
        while len(running_list) < max_running and len(to_be_run) > 0:
            runner = to_be_run.pop()
            print(runner.x, runner.y, runner.sbatch_nodes, runner.omp_threads)
            # Run the cconfig
            runner.run()
            running_list.append(runner)

        # extract the data once complete and add the columns which describe the config
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

            # Save the data to csv so generating data + plotting are separate
            df.to_csv(csv_path)
    #         all_df = pd.concat([all_df, df])
    #         print(all_df)
    # all_df.to_csv("timing.csv")

def plot_graphs():
    # Load all the data from the csv files
    filenames = glob.glob("timing_data/*.csv")
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)
    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Calculate the time taken for an entire timestep loop and save as a column
    all_df["loop_time_taken"] = all_df[["timestep_time_taken", "compute_velocity_time_taken", "rhs_time_taken", "possion_time_taken", "update_velocity_time_taken", "boundary_time_taken"]].sum(axis=1)

    # Run the plots
    plot_time_against_thread_count(all_df)
    plot_speed_up_against_thread_count(all_df)
    plot_speed_up_against_dimensions(all_df)
    plot_speed_up_against_sbatch_nodes(all_df)
    plot_parallel_efficiency_against_thread_count(all_df)
    plot_parallel_efficiency_against_sbatch_nodes(all_df)
    plot_parallel_efficiency_against_dimensions(all_df)

    plot_possion_loop_time_against_thread_count(all_df)
    plot_possion_loop_parallel_efficiency_against_thread_count(all_df)
    plot_sync_time_against_sbatch_nodes(all_df)
    plot_proportion_syncing_against_sbatch_nodes(all_df)


####################################################
#######         Plotting functions     #############
# 
# The following plotting functions all do similar things so I won't comment all of them
# The basic pattern is to group by configuration (nodes, threads, (x, y)) and get a mean of the time data
# Then iterate over two of the variables (creating the lines, e.g. nodes-threads) and the third forms the x axis
# The iterated two variables are used to select the subset of the data which is set to the values of the variables
# e.g. nodes = 1, threads = 4 selects the values where nodes == 1 and threads == 4, resulting in a vector of times for each dimension (third variable/ x axis)
# Plot the x axis values against the thing being measured (normally loop time taken)
# 
# Plotting parallel efficiency is done by extracting the ST time and doing ((ST / MT) / optimal speedup)
####################################################


def plot_time_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for sbatch_nodes, colour in zip(sbatch_nodes_plot, colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            plt.plot(node_df["omp_threads"], node_df["loop_time_taken"], colour + line_style, label=f"{sbatch_nodes} - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("OMP Threads")
    plt.ylabel("Time")
    plt.savefig("plots/time_against_thread_count.png", dpi=300)
    plt.clf()

def plot_speed_up_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        df_st_dim = df_st[df_st["x"] == x]
        df_st_dim = df_st_dim[df_st_dim["y"] == y]
        st_time_taken = df_st_dim["loop_time_taken"].iloc[0]
        print(st_time_taken)
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for sbatch_nodes, colour in zip(sbatch_nodes_plot, colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            plt.plot(node_df["omp_threads"], st_time_taken / node_df["loop_time_taken"], colour + line_style, label=f"{sbatch_nodes} - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("OMP Threads")
    plt.ylabel("Speed up over ST")
    plt.savefig("plots/speed_up_against_thread_count.png", dpi=300)
    plt.clf()

def plot_parallel_efficiency_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        df_st_dim = df_st[df_st["x"] == x]
        df_st_dim = df_st_dim[df_st_dim["y"] == y]
        st_time_taken = df_st_dim["loop_time_taken"].iloc[0]
        print(st_time_taken)
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for sbatch_nodes, colour in zip(sbatch_nodes_plot, colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            plt.plot(node_df["omp_threads"], (st_time_taken / node_df["loop_time_taken"]) / (node_df["omp_threads"] * sbatch_nodes), colour + line_style, label=f"{sbatch_nodes} - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("OMP Threads")
    plt.ylabel("Parallel Eff vs ST")
    plt.savefig("plots/parallel_efficiency_against_thread_count.png", dpi=300)
    plt.clf()

def plot_possion_loop_time_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    df_sn = df[df["sbatch_nodes"] == 1]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        dim_df = df_sn[df_sn["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        plt.plot(dim_df["omp_threads"], dim_df["possion_p_loop_time_taken"], "r" + line_style, label=f"p - {x}x{y}")
        plt.plot(dim_df["omp_threads"], dim_df["possion_res_loop_time_taken"], "b" + line_style, label=f"res - {x}x{y}")
        plt.plot(dim_df["omp_threads"], dim_df["compute_velocity_time_taken"], "g" + line_style, label=f"ctv - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("OMP Threads")
    plt.ylabel("Time (s)")
    plt.savefig("plots/possion_loop_time_against_thread_count.png", dpi=300)
    plt.clf()

def plot_possion_loop_parallel_efficiency_against_thread_count(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    df_sn = df[df["sbatch_nodes"] == 1]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        dim_df = df_sn[df_sn["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        df_st = dim_df[dim_df["omp_threads"] == 0]
        thread_nums = np.array(dim_df["omp_threads"]).copy()
        thread_nums[0] = 1
        plt.plot(dim_df["omp_threads"], (df_st["possion_p_loop_time_taken"].iloc[0] / dim_df["possion_p_loop_time_taken"]) / thread_nums, "r" + line_style, label=f"p - {x}x{y}")
        plt.plot(dim_df["omp_threads"], (df_st["possion_res_loop_time_taken"].iloc[0] / dim_df["possion_res_loop_time_taken"]) / thread_nums, "b" + line_style, label=f"res - {x}x{y}")
        plt.plot(dim_df["omp_threads"], (df_st["compute_velocity_time_taken"].iloc[0] / dim_df["compute_velocity_time_taken"]) / thread_nums, "g" + line_style, label=f"ctv - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("OMP Threads")
    plt.ylabel("Parallel Efficiency")
    plt.savefig("plots/possion_loop_parallel_efficiency_against_thread_count.png", dpi=300)
    plt.clf()

def plot_speed_up_against_dimensions(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":"]
    st_time_taken = np.array(df_st["loop_time_taken"])
    print(st_time_taken)
    fig, ax = plt.subplots(1,1)
    for omp_threads, line_style in zip(omp_num_threads_plot, line_styles):
        dim_df = df_par[df_par["omp_threads"] == omp_threads]
        for sbatch_nodes, colour in zip(sbatch_nodes_plot, colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            print(node_df["loop_time_taken"])
            plt.plot(range(1, len(dimensions) + 1), st_time_taken / np.array(node_df["loop_time_taken"]), colour + line_style, label=f"{sbatch_nodes}N - {omp_threads}T")

    plt.legend()
    plt.xticks(range(1, len(dimensions) + 1))
    ax.set_xticklabels([f"{x}x{y}" for (x,y) in dimensions])
    plt.xlabel("Dimensions")
    plt.ylabel("Speed up over ST")
    plt.savefig("plots/speed_up_against_dimension.png", dpi=300)
    plt.clf()

def plot_parallel_efficiency_against_dimensions(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":"]
    st_time_taken = np.array(df_st["loop_time_taken"])
    print(st_time_taken)
    fig, ax = plt.subplots(1,1)
    for omp_threads, line_style in zip(omp_num_threads_plot, line_styles):
        dim_df = df_par[df_par["omp_threads"] == omp_threads]
        for sbatch_nodes, colour in zip(sbatch_nodes_plot, colours):
            node_df = dim_df[dim_df["sbatch_nodes"] == sbatch_nodes]
            print(node_df["loop_time_taken"])
            plt.plot(range(1, len(dimensions) + 1), (st_time_taken / np.array(node_df["loop_time_taken"])) / (sbatch_nodes * omp_threads), colour + line_style, label=f"{sbatch_nodes}N - {omp_threads}T")

    plt.legend()
    plt.xticks(range(1, len(dimensions) + 1))
    ax.set_xticklabels([f"{x}x{y}" for (x,y) in dimensions])
    plt.xlabel("Dimensions")
    plt.ylabel("Parallel Eff vs ST")
    plt.savefig("plots/parallel_efficiency_against_dimension.png", dpi=300)
    plt.clf()

def plot_speed_up_against_sbatch_nodes(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        df_st_dim = df_st[df_st["x"] == x]
        df_st_dim = df_st_dim[df_st_dim["y"] == y]
        st_time_taken = df_st_dim["loop_time_taken"].iloc[0]
        print(st_time_taken)
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for omp_threads, colour in zip(omp_num_threads_plot, colours):
            node_df = dim_df[dim_df["omp_threads"] == omp_threads]
            plt.plot(node_df["sbatch_nodes"], st_time_taken / node_df["loop_time_taken"], colour + line_style, label=f"{omp_threads}T - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("Nodes")
    plt.ylabel("Speed up over ST")
    plt.savefig("plots/speed_up_against_sbatch_nodes.png", dpi=300)
    plt.clf()

def plot_parallel_efficiency_against_sbatch_nodes(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    df_st = df[df["omp_threads"] == 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        df_st_dim = df_st[df_st["x"] == x]
        df_st_dim = df_st_dim[df_st_dim["y"] == y]
        st_time_taken = df_st_dim["loop_time_taken"].iloc[0]
        print(st_time_taken)
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for omp_threads, colour in zip(omp_num_threads_plot, colours):
            node_df = dim_df[dim_df["omp_threads"] == omp_threads]
            plt.plot(node_df["sbatch_nodes"], (st_time_taken / node_df["loop_time_taken"]) / (omp_threads * node_df["sbatch_nodes"]), colour + line_style, label=f"{omp_threads}T - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("Nodes")
    plt.ylabel("Parallel Eff vs ST")
    plt.savefig("plots/parallel_efficiency_against_sbatch_nodes.png", dpi=300)
    plt.clf()

def plot_sync_time_against_sbatch_nodes(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for omp_threads, colour in zip(omp_num_threads_plot, colours):
            node_df = dim_df[dim_df["omp_threads"] == omp_threads]
            plt.plot(node_df["sbatch_nodes"], node_df["sync_time_taken"], colour + line_style, label=f"{omp_threads}T - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("Nodes")
    plt.ylabel("Time syncing (s)")
    plt.savefig("plots/sync_time_against_sbatch_nodes.png", dpi=300)
    plt.clf()

def plot_proportion_syncing_against_sbatch_nodes(all_df):
    df = all_df.groupby(["sbatch_nodes", "omp_threads", "x", "y"], as_index=False).mean()
    df_par = df[df["omp_threads"] > 0]
    # print(df)
    colours = ["r", "g", "b", "c", "m", "k", "y"]
    line_styles = ["-", "--", "-.", ":", ":"]
    for (x, y), line_style in zip(dimensions_plot, line_styles):
        dim_df = df_par[df_par["x"] == x]
        dim_df = dim_df[dim_df["y"] == y]
        for omp_threads, colour in zip(omp_num_threads_plot, colours):
            node_df = dim_df[dim_df["omp_threads"] == omp_threads]
            plt.plot(node_df["sbatch_nodes"], node_df["sync_time_taken"] / node_df["loop_time_taken"], colour + line_style, label=f"{omp_threads}T - {x}x{y}")

    plt.legend()
    plt.xticks(omp_num_threads_tested)
    plt.xlabel("Nodes")
    plt.ylabel("Proportion spent syncing")
    plt.savefig("plots/proportion_syncing_against_sbatch_nodes.png", dpi=300)
    plt.clf()

if __name__ == "__main__":
    # subprocess.run(["bash", "./clean_build.sh"])
    # collect_data()
    plot_graphs()
