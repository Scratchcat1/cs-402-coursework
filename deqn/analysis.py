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

def run_deqn(environment_variables = {}):
    process_output = subprocess.run(["bash", "./run_local.sh"], capture_output=True, env={**os.environ, **environment_variables})
    output_lines = process_output.stdout.decode().split("\n")
    # print(process_output.stderr)
    timing_lines = [line for line in output_lines if "seconds" in line]
    
    timing_results = []
    assert len(timing_lines) % 3 == 0
    for i in range(0, len(timing_lines), 3):
        diffuse_time = get_time_from_timing_line(timing_lines[i])
        reset_time = get_time_from_timing_line(timing_lines[i + 1])
        update_boundaries_time = get_time_from_timing_line(timing_lines[i + 2])
        timing_results.append([i // 3, diffuse_time, reset_time, update_boundaries_time])

    # print(timing_results)
    df = pd.DataFrame(timing_results, columns=("Iteration", "Diffuse", "Reset", "Update Boundaries"))
    # print(df)
    return df

def time_against_thread_count():
    df = pd.DataFrame({
        "Thread Count": pd.Series(dtype='int32'),
        "Iteration": pd.Series(dtype='int32'),
        "Diffuse": pd.Series(dtype='float'),
        "Reset": pd.Series(dtype='float'),
        "Update Boundaries": pd.Series(dtype='float')
        })
    for thread_count in [1, 2, 4, 8, 12, 16, 32]:
        df_for_thread_count = run_deqn({"OPENMP_NUM_THREADS":str(thread_count)})
        df_for_thread_count["Thread Count"] = thread_count
        df = pd.concat([df, df_for_thread_count], ignore_index = True)
    print(df)
    thread_means_df = df.groupby(["Thread Count"]).mean()
    x = np.arange(len(thread_means_df))
    width = 0.2
    # plt.boxplot(thread_means_df["Diffuse"], vert=True)
    # plt.boxplot(thread_means_df["Reset"], vert=True)
    # plt.boxplot(thread_means_df["Update Boundaries"], vert=True)
    print(df.dtypes)
    df = df.drop(columns=["Iteration"])
    df.boxplot(by="Thread Count", showfliers=False, layout=(3, 1))
    plt.savefig("plots/time_against_thread_count.png")
    plt.show()

if __name__ == "__main__":
    run_deqn({"OPENMP_NUM_THREADS":"2"})
    time_against_thread_count()