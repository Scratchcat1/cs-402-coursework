#!/usr/bin/python3
import pandas as pd
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_rows', None)

def get_time_from_timing_line(line):
    string_time = line.split(" ")[3]
    return float(string_time)

class DeqnConfigFile:
    def __init__(self):
        self.nx = 1000
        self.ny = 1000
        self.xmin = 0.0
        self.ymin = 0.0
        self.xmax = 1000.0
        self.ymax = 1000.0
        self.initial_dt = 0.04
        self.end_time = 1.80
        self.scheme = "explicit"
        self.vis_frequency = -1
        self.subregion = (30.0, 30.0, 60.0, 60.0)
    
    def save_to_file(self, filepath):
        with open(filepath, "w") as fh:
            fh.writelines([
                f"nx {self.nx}\n",
                f"ny {self.ny}\n",
                f"xmin {self.xmin}\n",
                f"ymin {self.ymin}\n",
                f"xmax {self.xmax}\n",
                f"ymax {self.ymax}\n",
                f"initial_dt {self.initial_dt}\n",
                f"end_time {self.end_time}\n",
                f"scheme {self.scheme}\n",
                f"vis_frequency {self.vis_frequency}\n",
                f"subregion {self.subregion[0]} {self.subregion[1]} {self.subregion[2]} {self.subregion[3]}\n",
            ])

def run_deqn(filepath, environment_variables = {}):
    process_output = subprocess.run(["bash", "./run_for_python.sh", filepath], stdout=subprocess.PIPE, env={**os.environ, **environment_variables})
    output_lines = process_output.stdout.decode().split("\n")
#    print(process_output.stdout)
    timing_lines = [line for line in output_lines if "seconds" in line]
    
    timing_results = []
    assert len(timing_lines) % 3 == 0
    for i in range(0, len(timing_lines), 3):
        diffuse_time = get_time_from_timing_line(timing_lines[i])
        reset_time = get_time_from_timing_line(timing_lines[i + 1])
        update_boundaries_time = get_time_from_timing_line(timing_lines[i + 2])
        timing_results.append([i // 3, diffuse_time, reset_time, update_boundaries_time, diffuse_time + reset_time + update_boundaries_time])

    # print(timing_results)
    df = pd.DataFrame(timing_results, columns=("Iteration", "Diffuse", "Reset", "Update Boundaries", "Total"))
    # print(df)
    return df

def time_against_thread_count_by_function():
    print("time_against_thread_count_by_function")
    df = pd.DataFrame({
        "Thread Count": pd.Series(dtype='int32'),
        "Iteration": pd.Series(dtype='int32'),
        "Diffuse": pd.Series(dtype='float'),
        "Reset": pd.Series(dtype='float'),
        "Update Boundaries": pd.Series(dtype='float'),
        "Total": pd.Series(dtype='float')
        })
    deqn_config_filepath = os.path.join("test", "tmp_square_by_threadcount.in")
    square_size = 10000

    for thread_count in [0, 1, 2, 4, 8, 12, 16, 32]:
        deqn_config_file = DeqnConfigFile()
        deqn_config_file.nx = square_size
        deqn_config_file.ny = square_size
        deqn_config_file.xmax = float(square_size)
        deqn_config_file.ymax = float(square_size)
        if thread_count == 0:
            deqn_config_file.scheme = "explicit_single_thread"
        deqn_config_file.save_to_file(deqn_config_filepath)

        df_for_thread_count = run_deqn(os.path.join(os.path.pardir, deqn_config_filepath), {"OMP_NUM_THREADS":str(max(1, thread_count))})
        df_for_thread_count["Thread Count"] = thread_count
        df = pd.concat([df, df_for_thread_count], ignore_index = True)
    print(df)
    thread_means_df = df.groupby(["Thread Count"]).median()
    x = np.arange(len(thread_means_df))
    width = 0.2
    # plt.boxplot(thread_means_df["Diffuse"], vert=True)
    # plt.boxplot(thread_means_df["Reset"], vert=True)
    # plt.boxplot(thread_means_df["Update Boundaries"], vert=True)
    print(df.dtypes)
    df = df.drop(columns=["Iteration"])
    for attribute in ["Diffuse", "Reset", "Update Boundaries", "Total"]:
        df[["Thread Count", attribute]].boxplot(by="Thread Count", showfliers=False)
        plt.savefig(f"plots/time_against_thread_count_attribute_f{attribute.lower().replace(' ', '_')}.png")
        plt.clf()

def time_against_square_size_by_thread_count():
    print("time_against_square_size_by_thread_count")
    df = pd.DataFrame({
        "Square Size": pd.Series(dtype='int32'),
        "Thread Count": pd.Series(dtype='int32'),
        "Iteration": pd.Series(dtype='int32'),
        "Diffuse": pd.Series(dtype='float'),
        "Reset": pd.Series(dtype='float'),
        "Update Boundaries": pd.Series(dtype='float'),
        "Total": pd.Series(dtype='float')
        })
    deqn_config_filepath = os.path.join("test", "tmp_square_by_threadcount.in")
    
    square_sizes = np.array([2**x for x in range(3, 14)])
    thread_counts = [0, 1, 2, 4, 8, 12]
    for square_size in square_sizes:
        for thread_count in thread_counts:
            # Generate the config file
            print("Square size: ", square_size, "  Thread count: ", thread_count)
            deqn_config_file = DeqnConfigFile()
            deqn_config_file.nx = square_size
            deqn_config_file.ny = square_size
            deqn_config_file.xmax = float(square_size)
            deqn_config_file.ymax = float(square_size)
            if thread_count == 0:
                deqn_config_file.scheme = "explicit_single_thread"
            deqn_config_file.save_to_file(deqn_config_filepath)

            df_for_thread_count = run_deqn(os.path.join(os.path.pardir, deqn_config_filepath), {"OMP_NUM_THREADS":str(max(1, thread_count))})
            df_for_thread_count["Square Size"] = square_size
            df_for_thread_count["Thread Count"] = thread_count
            df = pd.concat([df, df_for_thread_count], ignore_index = True)
    df_by_size = df.groupby(["Square Size", "Thread Count"], as_index=False).median()

    # Timing graph
    for thread_count in thread_counts:
        plt.plot(square_sizes, df_by_size[df_by_size["Thread Count"] == thread_count]["Total"], label=f"{thread_count}T")
    plt.xlabel("Square edge length")
    plt.ylabel("Total time (microseconds)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"plots/time_against_square_size_by_thread_count.png")
    plt.clf()

    # Speed up graph
    for thread_count in thread_counts:
        proportional_speedup = np.array(df_by_size[df_by_size["Thread Count"] == 0]["Total"]) / np.array(df_by_size[df_by_size["Thread Count"] == thread_count]["Total"])
        plt.plot(square_sizes, proportional_speedup, label=f"{thread_count}T")
    plt.xlabel("Square edge length")
    plt.ylabel("Proportional speedup (vs ST)")
    plt.legend()
    plt.xscale('log')
    plt.savefig(f"plots/speed_up_against_square_size_by_thread_count.png")
    plt.clf()

    # Memory bandwidth
    for thread_count in thread_counts:
        current_thread_count_data = df_by_size[df_by_size["Thread Count"] == thread_count]
        memory_bandwidth = 2 * 8 * np.square(np.array(current_thread_count_data["Square Size"])) / (current_thread_count_data["Total"] / 1e6)
        plt.plot(square_sizes, memory_bandwidth, label=f"{thread_count}T")
    plt.xlabel("Square edge length")
    plt.ylabel("Memory bandwidth (B/s)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"plots/memory_bandwidth_against_square_size_by_thread_count.png")
    plt.clf()

def time_against_square_size_by_tile_size():
    print("time_against_square_size_by_tile_size")
    df = pd.DataFrame({
        "Square Size": pd.Series(dtype='int32'),
        "Tile Size": pd.Series(dtype='int32'),
        "Iteration": pd.Series(dtype='int32'),
        "Diffuse": pd.Series(dtype='float'),
        "Reset": pd.Series(dtype='float'),
        "Update Boundaries": pd.Series(dtype='float'),
        "Total": pd.Series(dtype='float')
        })
    deqn_config_filepath = os.path.join("test", "tmp_square_by_threadcount.in")
    
    square_sizes = np.array([2**x for x in range(3, 14)])
    tile_sizes = [0, 4, 8, 16, 32, 64, 128, 256, 512]
    for square_size in square_sizes:
        for tile_size in tile_sizes:
            # Generate the config file
            print("Square size: ", square_size, "  Tile Size: ", tile_size)
            deqn_config_file = DeqnConfigFile()
            deqn_config_file.nx = square_size
            deqn_config_file.ny = square_size
            deqn_config_file.xmax = float(square_size)
            deqn_config_file.ymax = float(square_size)
            if tile_size == 0:
                deqn_config_file.scheme = "explicit"
            else:
                deqn_config_file.scheme = "explicit_tiles"
            deqn_config_file.save_to_file(deqn_config_filepath)

            df_for_tile_size = run_deqn(os.path.join(os.path.pardir, deqn_config_filepath))
            df_for_tile_size["Square Size"] = square_size
            df_for_tile_size["Tile Size"] = tile_size
            df = pd.concat([df, df_for_tile_size], ignore_index = True)
    df_by_size = df.groupby(["Square Size", "Tile Size"], as_index=False).median()

    # Timing graph
    for tile_size in tile_sizes:
        plt.plot(square_sizes, df_by_size[df_by_size["Tile Size"] == tile_size]["Total"], label=f"{tile_size}x{tile_size}")
    plt.xlabel("Square edge length")
    plt.ylabel("Total time (microseconds)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"plots/time_against_square_size_by_tile_size.png")
    plt.clf()

    # Speed up graph
    for tile_size in tile_sizes:
        proportional_speedup = np.array(df_by_size[df_by_size["Tile Size"] == 0]["Total"]) / np.array(df_by_size[df_by_size["Tile Size"] == tile_size]["Total"])
        plt.plot(square_sizes, proportional_speedup, label=f"{tile_size}x{tile_size}")
    plt.xlabel("Square edge length")
    plt.ylabel("Proportional speedup (vs NoTilesMT)")
    plt.legend()
    plt.xscale('log')
    plt.savefig(f"plots/speed_up_against_square_size_by_tile_size.png")
    plt.clf()

    # Memory bandwidth
    for tile_size in tile_sizes:
        current_tile_size_data = df_by_size[df_by_size["Tile Size"] == tile_size]
        memory_bandwidth = 2 * 8 * np.square(np.array(current_tile_size_data["Square Size"])) / (current_tile_size_data["Total"] / 1e6)
        plt.plot(square_sizes, memory_bandwidth, label=f"{tile_size}x{tile_size}")
    plt.xlabel("Square edge length")
    plt.ylabel("Memory bandwidth (B/s)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"plots/memory_bandwidth_against_square_size_by_tile_size.png")
    plt.clf()

def time_against_square_size_by_schedule():
    print("time_against_square_size_by_schedule")
    df = pd.DataFrame({
        "Square Size": pd.Series(dtype='int32'),
        "Schedule": pd.Series(dtype=object),
        "Iteration": pd.Series(dtype='int32'),
        "Diffuse": pd.Series(dtype='float'),
        "Reset": pd.Series(dtype='float'),
        "Update Boundaries": pd.Series(dtype='float'),
        "Total": pd.Series(dtype='float')
        })
    deqn_config_filepath = os.path.join("test", "tmp_square_by_threadcount.in")
    
    square_sizes = np.array([2**x for x in range(3, 14)])
    schedules = ["static", "dynamic,1", "dynamic,8", "guided", "guided,8", "auto"]
    for square_size in square_sizes:
        for schedule in schedules:
            # Generate the config file
            print("Square size: ", square_size, "  Schedule: ", schedule)
            deqn_config_file = DeqnConfigFile()
            deqn_config_file.nx = square_size
            deqn_config_file.ny = square_size
            deqn_config_file.xmax = float(square_size)
            deqn_config_file.ymax = float(square_size)
            deqn_config_file.scheme = "explicit_runtime_schedule"
            deqn_config_file.save_to_file(deqn_config_filepath)

            df_for_schedule = run_deqn(os.path.join(os.path.pardir, deqn_config_filepath), {"OMP_SCHEDULE": schedule})
            df_for_schedule["Square Size"] = square_size
            df_for_schedule["Schedule"] = schedule
            df = pd.concat([df, df_for_schedule], ignore_index = True)
    df_by_size = df.groupby(["Square Size", "Schedule"], as_index=False).median()

    # Timing graph
    for schedule in schedules:
        plt.plot(square_sizes, df_by_size[df_by_size["Schedule"] == schedule]["Total"], label=f"{schedule}")
    plt.xlabel("Square edge length")
    plt.ylabel("Total time (microseconds)")
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f"plots/time_against_square_size_by_schedule.png")
    plt.clf()

if __name__ == "__main__":
    subprocess.run(["bash", "./clean_build.sh"])
    # run_deqn({"OMP_NUM_THREADS":"2"})
    # time_against_thread_count_by_function()
    # time_against_square_size_by_thread_count()
    # time_against_square_size_by_tile_size()
    time_against_square_size_by_schedule()
