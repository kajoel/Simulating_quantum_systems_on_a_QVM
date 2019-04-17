"""
Test of parallel gain. This script has to be run from a terminal (at least it
doesn't work for me in PyCharm).

@author = Joel
"""

from core.data import save, load
from os.path import join
from socket import gethostname
import subprocess
from time import perf_counter
from constants import ROOT_DIR

# Path to the script to run
script_path = join(ROOT_DIR, 'misc', 'meas_useless.py')

directory = 'parallel_test'  # directory to save to
file = 'parallel_test_results_2'  # file to save to

# Complete path to the saved file (relative to the data directory):
path = join(directory, file)

try:
    # Try to load the file (will raise FileNotFoundError if not existing)
    data, metadata = load(path)
except FileNotFoundError:
    # Initialize data and metadata
    data = {'workers': [], 'time': []}
    metadata = {'description': "File containing results from parallel test. "
                               "'workers' is the number of parallel instances "
                               "and 'time' is the average time it took for "
                               "them to complete (i.e. total time/workers). "
                               "Data from different files are not comparable.",
                'machine': gethostname(),
                'count': 0}


# Loop trough some parameters with a counter.
count = 0
for num_workers in [i+1 for i in range(20)]*10:
    count += 1
    if count > metadata['count']:
        # In this case the relevant simulation has not already been saved to the
        # file and is therefore run now.

        start_time = perf_counter()

        # Start workers
        workers = []
        for i in range(num_workers):
            workers.append(subprocess.Popen(['python', script_path]))

        # Wait for workers to finish
        for worker in workers:
            worker.wait()

        stop_time = perf_counter()

        data['workers'].append(num_workers)
        data['time'].append((stop_time-start_time)/num_workers)

        # Update metadata['count'] and save file:
        metadata['count'] = count
        save(path, data, metadata)
