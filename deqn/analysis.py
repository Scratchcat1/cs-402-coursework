#!/usr/bin/python3
import pandas as pd
import subprocess

def get_time_from_timing_line(line):
    string_time = line.split(" ")[3]
    return float(string_time)

def run_deqn(environment_variables = []):
    process_output = subprocess.run([*environment_variables, "bash", "./run_local.sh"], capture_output=True)
    output_lines = process_output.stdout.decode().split("\n")
    timing_lines = [line for line in output_lines if "seconds" in line]
    
    timing_results = []
    assert len(timing_lines) % 3 == 0
    for i in range(0, len(timing_lines), 3):
        diffuse_time = get_time_from_timing_line(timing_lines[i])
        reset_time = get_time_from_timing_line(timing_lines[i + 1])
        update_boundaries_time = get_time_from_timing_line(timing_lines[i + 2])
        timing_results.append([diffuse_time, reset_time, update_boundaries_time])

    print(timing_results)
    df = pd.DataFrame(timing_results, columns=("Diffuse", "Reset", "Update Boundaries"))
    print(df)


if __name__ == "__main__":
    run_deqn()