"""
Module with function for running in parallel and doing cleanups after runs etc.
"""
import numpy as np
from time import perf_counter
import os
from os.path import join, isfile
from multiprocessing import Pool
from uuid import getnode as get_mac
from core import data
from constants import ROOT_DIR


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


def run(simulate,
        identifier_generator,
        input_functions,
        directory,
        version,
        metametadata,
        file_from_id,
        metadata_from_id,
        num_workers=os.cpu_count(),
        start_range=0,
        stop_range=np.inf,
        max_task=1,
        chunksize=1):
    """
    Run simulate in parallel.

    :return:
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
    def wrap(x):
        # Use a broad try-except to not crash if we don't have to
        try:
            return x[0], simulate(*x[0], *x[1])

        except Exception as e:
            # This will be saved in the metadata file.
            return x[0], e

    # Generator for pool
    generator = Bookkeeper(identifier_generator, ids, input_functions,
                           [start_range, stop_range])

    # Counters and such
    max_num_workers = os.cpu_count()
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
                    file_ = file_from_id(identifier)
                    if file_ not in files:
                        files.add(file_)
                        if not isfile(join(base_dir, directory,
                                           file_ + '.pkl')):
                            # Create file
                            metadata = metadata_from_id(identifier)
                            data.save(join(directory, file_), [], metadata,
                                      extract=True, base_dir=base_dir)

                    data.append(join(directory, file_), [identifier, result],
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
        meta_dict = {}
        for x in metadata:
            # Keep only the last exception for given identifier.
            if x[0] not in meta_dict or meta_dict[x[0]] is not True:
                meta_dict[x[0]] = x[1]
        metadata = [[x, meta_dict[x]] for x in meta_dict]
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
        done = sum(x[1] for x in metadata if x[1] is True)
        print(f'Number of tasks remaining: {len(metadata) - done}')
