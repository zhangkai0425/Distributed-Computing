import subprocess
import re

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, _ = process.communicate()
    return stdout

num_runs = 10
serial_times = []
threads_times = []

for _ in range(num_runs):
    output = run_command('srun -n 1 ./pr /data/hw2_data/com-orkut_117m.graph 64')
    
    serial_match = re.search(r'Serial Reference Page Rank\n\s+\d+:\s+([\d.]+) s', output)
    threads_match = re.search(r'Threads  Page Rank\n\s+\d+:\s+([\d.]+) s', output)
    
    if serial_match and threads_match:
        serial_time = float(serial_match.group(1))
        threads_time = float(threads_match.group(1))
        serial_times.append(serial_time)
        threads_times.append(threads_time)

average_serial_time = sum(serial_times) / num_runs
average_threads_time = sum(threads_times) / num_runs

print(f'Average Serial Time: {average_serial_time} s')
print(f'Average Threads Time: {average_threads_time} s')
print(f'Average Speed up : {average_serial_time/average_threads_time} s')