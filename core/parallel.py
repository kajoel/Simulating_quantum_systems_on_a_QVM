"""
Module with function for running in parallel and doing cleanups after runs etc.

@author = Joel
"""
import numpy as np
import warnings
import os
from os.path import join, isfile
from multiprocessing import Pool
from time import perf_counter, sleep
from uuid import getnode as get_mac
from core import data
from constants import ROOT_DIR

max_num_workers = os.cpu_count()


class Wrap:
    """
    Class for wrapping simulate (need pickable object for multiprocess Pool)
    """
    def __init__(self, simulate):
        self.simulate = simulate

    def __call__(self, x):
        # Use a broad try-except to not crash if we don't have to
        try:
            return x[0], self.simulate(*x[0], *x[1])

        except Exception as e:
            # This will be saved in the metadata file.
            return x[0], e


class Bookkeeper:
    """
    Class for keeping track of what's been done and only assign new tasks.
    """

    def __init__(self, iterator, book, output_calc=None, bounds=None):
        """

        :param iterator: Iterable
        :param set book: Set of identifiers corresponding to previously
            completed tasks.
        :param output_calc: List of functions
        :param bounds: Bounds of elements to return from iterator
        """
        if output_calc is None:
            output_calc = {}
        if bounds is None:
            bounds = [0, np.inf]
        self.iterator = iterator
        self.book = book
        self.output_calc = output_calc
        self.bounds = bounds
        self.count = -1  # to start from 0 (see += 1 below)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x = self.iterator.__next__()
            self.count += 1

            if self.count >= self.bounds[1]:
                raise StopIteration

            if x not in self.book and self.count >= self.bounds[0]:
                output = []
                for i in range(len(x) + 1):
                    if i in self.output_calc:
                        output.extend(self.output_calc[i](
                            *[y for j, y in enumerate(x) if j < i]))
                return x, output


def script_input(args):
    """
    Parse script inputs to parallel.run

    :param args: Input sys.argv
    :return: kwargs to parallel.run
    """
    # Input number of workers
    if len(args) <= 1:
        num_workers = max_num_workers
    else:
        try:
            num_workers = int(args[1])
        except ValueError:
            num_workers = max_num_workers
            warnings.warn(f'Could not parse input parameter 1 num_workers.')

    # Input start of range
    if len(args) <= 2:
        start_range = 0
    else:
        try:
            start_range = int(args[2])
        except ValueError:
            start_range = 0
            warnings.warn(f'Could not parse input parameter 2 start_range.')

    # Input end of range
    if len(args) <= 3:
        stop_range = np.inf
    else:
        try:
            stop_range = int(args[3])
        except ValueError:
            stop_range = np.inf
            warnings.warn(f'Could not parse input parameter 3 stop_range.')

    print(f'\nStarting with num_workers = {num_workers}, and range = '
          f'[{start_range}, {stop_range})\n')

    return {'num_workers': num_workers,
            'start_range': start_range,
            'stop_range': stop_range}


def run(simulate,
        identifier_generator,
        input_functions,
        directory,
        version,
        script_file,
        file_from_id,
        metadata_from_id,
        num_workers=os.cpu_count(),
        start_range=0,
        stop_range=np.inf,
        max_task=1,
        chunksize=1,
        restart=True,
        delay=5):
    """
    # TODO: param doc

    :param simulate:
    :param identifier_generator:
    :param input_functions:
    :param directory:
    :param version:
    :param script_file:
    :param file_from_id:
    :param metadata_from_id:
    :param num_workers:
    :param start_range:
    :param stop_range:
    :param max_task:
    :param chunksize:
    :param restart:
    :param delay:
    :return:
    """
    remaining = 1
    while restart and remaining:
        remaining = _run_internal(
            simulate,
            identifier_generator(),
            input_functions,
            directory,
            version,
            script_file,
            file_from_id,
            metadata_from_id,
            num_workers,
            start_range,
            stop_range,
            max_task,
            chunksize,
            )
        if remaining:
            print(f'\n{remaining} tasks remaining.\nRestarting in {delay} s.\n')
            sleep(delay)


def _run_internal(simulate,
                  identifier_generator,
                  input_functions,
                  directory,
                  version,
                  script_file,
                  file_from_id,
                  metadata_from_id,
                  num_workers,
                  start_range,
                  stop_range,
                  max_task,
                  chunksize):
    """
    Internal parallel run.

    :return: number of remaining tasks.
    """
    # Files and paths
    base_dir = join(ROOT_DIR, 'data_ignore')
    directory += f'_v{version}'
    directory = join(directory, str(get_mac()))
    path_metadata = join(directory, 'metadata')
    try:
        # Try to load the file (will raise FileNotFoundError if not existing)
        metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
    except FileNotFoundError:
        metadata = []
        metametadata = {
            'description': "File that keeps track of what's been done "
                           "previously in this script "
            f"({script_file}).",
            'run_time': [],
            'num_workers': [],
            'num_tasks_completed': [],
            'success_rate': [],
            'created_from': script_file}
        data.save(file=path_metadata, data=metadata, metadata=metametadata,
                  extract=True, base_dir=base_dir)

    # Extract identifiers to previously completed simulations
    ids = set()
    for id_, value in metadata:
        if value is True:
            if id_ in ids:
                raise KeyError('Multiple entries for same id in metadata.')
            ids.add(id_)

    # Cleanup (metadata can contain thousands of Exception objects)
    del metadata, metametadata

    # Wrap simulate to get expected input/output and handle exceptions
    wrap = Wrap(simulate)

    # Generator for pool
    generator = Bookkeeper(identifier_generator, ids, input_functions,
                           [start_range, stop_range])

    # Counters and such
    files = set()
    success = 0
    fail = 0
    start_time = perf_counter()

    # Actual run
    try:
        with Pool(num_workers, maxtasksperchild=max_task) as p:
            result_generator = p.imap_unordered(wrap, generator,
                                                chunksize=chunksize)
            for identifier, result in result_generator:
                # Handle exceptions:
                if isinstance(result, Exception):
                    fail += 1
                    # Save the error
                    data.append(path_metadata, [identifier, result],
                                base_dir=base_dir)
                else:
                    success += 1
                    file = file_from_id(identifier)
                    if file not in files:
                        files.add(file)
                        if not isfile(join(base_dir, directory,
                                           file + '.pkl')):
                            # Create file
                            metadata = metadata_from_id(identifier)
                            data.save(join(directory, file), [], metadata,
                                      extract=True, base_dir=base_dir)

                    data.append(join(directory, file), [identifier, result],
                                base_dir=base_dir)

                    # Mark the task as completed (last in the else,
                    # after saving result)
                    data.append(path_metadata, [identifier, True],
                                base_dir=base_dir)
    finally:
        stop_time = perf_counter()
        if success + fail == 0:
            fail = 1  # To avoid division by zero

        # Post simulation.
        metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
        metadata = cleanup_small(metadata)
        metametadata['run_time'].append(stop_time - start_time)
        metametadata['num_workers'].append((num_workers, max_num_workers))
        metametadata['num_tasks_completed'].append(success)
        metametadata['success_rate'].append(success / (success + fail))
        data.save(file=path_metadata, data=metadata, metadata=metametadata,
                  extract=True, base_dir=base_dir)

        # Print some stats
        print('\nSimulation completed.')
        print(f'Total number of tasks this far: {len(metadata)}')
        print(f'Completed tasks this run: {success}')
        print(f'Success rate this run: {success / (success + fail)}')
        remaining = len(metadata) - sum(x[1] for x in metadata if x[1] is True)
        print(f'Number of tasks remaining: {remaining}')

        # Return remaining for restart
        return remaining


def cleanup_small(metadata):
    """
    Cleanup metadata at path.

    :param metadata: unclean metadata
    :return: cleaned metadata
    """
    meta_dict = {}
    for x in metadata:
        # Keep only the last exception for given identifier.
        if x[0] not in meta_dict or meta_dict[x[0]] is not True:
            meta_dict[x[0]] = x[1]
    return [[x, meta_dict[x]] for x in meta_dict]
