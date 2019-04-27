"""
Module with function for running in parallel and doing cleanups after runs etc.

BE AWARE: This module is not completely safe for use on multiple computers
with shared file system. In particular, it's not safe to start init or
cleanup at the same time as anything else.

@author = Joel
"""
import numpy as np
import warnings
import shutil
import os
from os.path import join, isfile
from multiprocessing import Pool
from time import perf_counter, sleep
from datetime import datetime
from uuid import getnode as get_mac
from pathlib import Path
from core import data
from constants import ROOT_DIR

max_num_workers = os.cpu_count()
base_dir = join(ROOT_DIR, 'data_ignore')
mac = get_mac()


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
    # TODO: better script_input that can take string 'cleanup' and 'init'

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
        num_workers=max_num_workers,
        start_range=0,
        stop_range=np.inf,
        max_task=1,
        chunksize=1,
        restart=True,
        delay=5,
        init=False,
        cleanup=False):
    """
    Run simulations, metadata initialization or cleanup.
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
    :param init:
    :param cleanup:
    :return:
    """
    directory = join(directory, f'v{version}')
    task = 'run'*(not init and not cleanup) \
           + 'cleanup'*(not init and cleanup) \
           + 'init'*init
    _mark_running(directory, task)
    try:
        if init:
            if _is_running(directory, ('run', 'cleanup', 'init')):
                raise RuntimeError("Can't run init when anyone else is "
                                   "running.")
            _init_metadata(identifier_generator(),
                           directory,
                           script_file)
        elif cleanup:
            if _is_running(directory, ('run', 'cleanup', 'init')):
                raise RuntimeError("Can't run cleanup when anyone else is "
                                   "running.")
            _cleanup_big(identifier_generator(), directory, script_file)
        else:
            if _is_running(directory, ('cleanup', 'init')):
                raise RuntimeError("Can't run simulation when someone else is "
                                   "initializing metadata or cleaning up.")
            while restart and _run_internal(
                    simulate,
                    identifier_generator(),
                    input_functions,
                    directory,
                    script_file,
                    file_from_id,
                    metadata_from_id,
                    num_workers,
                    start_range,
                    stop_range,
                    max_task,
                    chunksize,
                    ):
                print(f'\nRestarting in {delay} s.\n')
                sleep(delay)
    finally:
        _mark_not_running(directory, task)


def _run_internal(simulate,
                  identifier_generator,
                  input_functions,
                  directory,
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
    directory_nomac = directory
    directory = join(directory, str(mac))
    path_metadata = join(directory, 'metadata')
    try:
        # Try to load the file (will raise FileNotFoundError if not existing)
        metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
    except FileNotFoundError:
        metadata = _get_metadata(directory_nomac)[0]
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
        total = success + fail
        if total == 0:
            total = 1  # To avoid division by zero

        # Post simulation.
        metadata, metametadata = data.load(path_metadata, base_dir=base_dir)
        metadata = _cleanup_small(metadata)
        metametadata['run_time'].append(stop_time - start_time)
        metametadata['num_workers'].append((num_workers, max_num_workers))
        metametadata['num_tasks_completed'].append(success)
        metametadata['success_rate'].append(success / total)
        data.save(file=path_metadata, data=metadata, metadata=metametadata,
                  extract=True, base_dir=base_dir)

        # Print some stats
        print('\nSimulation completed.')
        print(f'Total number of tasks this far: {len(metadata)}')
        print(f'Completed tasks this run: {success}')
        print(f'Success rate this run: {success / total}')
        remaining = len(metadata) - sum(x[1] for x in metadata if x[1] is True)
        print(f'Minimum number of tasks remaining for this run: {fail}')
        print(f'Total number of tasks remaining: {remaining}')

        # No success and no fail => no restart
        return success + fail


def _init_metadata(identifier_generator, directory, script_file, force=False):
    """
    Initialize metadata.

    :param identifier_generator: id generator
    :param directory: directory
    :param script_file: name of script
    :param force: if False raise RuntimeError if metadata is already existing.
    :return:
    """
    if not force and _get_metadata(directory, warn=False) != ([], {}):
        raise RuntimeError('Metadata has already been initialized.')
    # Fix directory and path
    directory = join(directory, 'total')
    path_metadata = join(directory, 'metadata')

    # Save metadata file
    metadata = []
    metametadata = {
        'description': "File that keeps track of what's been done "
                       "previously in this script "
                       f"({script_file}).",
        'created_from': script_file}
    data.save(file=path_metadata, data=metadata, metadata=metametadata,
              extract=True, base_dir=base_dir)

    # Add identifiers
    count = 0
    start_time = perf_counter()
    for identifier in identifier_generator:
        count += 1
        data.append(path_metadata, [identifier, False], base_dir=base_dir)
    stop_time = perf_counter()

    print(f'\nMetadata initialization completed in'
          f'{stop_time - start_time: .1f} s with {count} identifiers.\n')


def _cleanup_big(identifier_generator, directory, script_file):
    """
    Cleanup directory by going trough all subdirectories, collect results and
    fix metadata.

    :param directory: Directory of cleanup.
    :return:
    """
    start_time = perf_counter()
    # Get total/metadata
    metadata, metametadata = _get_metadata(directory, warn=False)
    if (metadata, metametadata) == ([], {}):
        _init_metadata(identifier_generator, directory, script_file)

        # Try again
        metadata, metametadata = _get_metadata(directory, warn=False)
        if (metadata, metametadata) == ([], {}):
            raise ValueError("_init_metadata doesn't work")

    # Convert metadata to dict
    meta_dict = {}
    for x in metadata:
        # Keep only the last exception for given identifier.
        if x[0] not in meta_dict or meta_dict[x[0]] is not True:
            meta_dict[x[0]] = x[1]
    del metadata

    # Find mac-subdirectories
    subdirs = set()
    with os.scandir(join(base_dir, directory)) as it:
        for entry in it:
            if entry.is_dir() and entry.name.isdigit():
                subdirs.add(entry.name)

    # Find data-files and keep track of which exists in which subdir
    files = {}
    for subdir in subdirs:
        with os.scandir(join(base_dir, directory, subdir)) as it:
            for entry in it:
                if entry.is_file() and entry.name != 'metadata.pkl':
                    if entry.name not in files:
                        files[entry.name] = []
                    files[entry.name].append(subdir)

    # Go through files, create file in total (if not existing), add data
    # from files in subdirs and update meta_dict
    count = 0
    for file in files:
        # Load file from total
        try:
            content, metadata = data.load(file=join(directory, 'total', file),
                                          base_dir=base_dir)
        except FileNotFoundError:
            content, metadata = [], {}

        # Convert content to dict
        content_dict = _add_result_to_dict(content, {})
        del content

        # Add content from other files
        for subdir in files[file]:
            content_new, metadata_new = data.load(
                file=join(directory, subdir, file), base_dir=base_dir)

            content_dict = _add_result_to_dict(content_new, content_dict)

            # Change metadata if no previous
            if metadata == {}:
                metadata = metadata_new
                metadata['created_by'] = data.get_name()
                metadata['created_datetime'] = datetime.now().\
                    strftime("%Y-%m-%d, %H:%M:%S")

        # Convert content_dict back to list and update meta_dict
        content = []
        for id_ in content_dict:
            if id_ not in meta_dict or meta_dict[id_] is not True:
                meta_dict[id_] = True

            for result in content_dict[id_]:
                count += 1
                content.append([id_, result])

        # Save file
        data.save(file=join(directory, 'total', file), data=content,
                  metadata=metadata, extract=True, base_dir=base_dir,
                  disp=False)
    del metadata

    # Convert meta_dict back to list and save
    metadata = []
    for id_ in meta_dict:
        metadata.append([id_, meta_dict[id_]])

    data.save(file=join(directory, 'total', 'metadata'), base_dir=base_dir,
              data=metadata, metadata=metametadata, extract=True, disp=False)

    # Update metadata in subdirs
    for subdir in subdirs:
        try:
            metametadata_new = data.load(
                file=join(directory, subdir, 'metadata'), base_dir=base_dir)[1]
        except FileNotFoundError:
            metametadata_new = metametadata

        data.save(file=join(directory, subdir, 'metadata'), base_dir=base_dir,
                  data=metadata, metadata=metametadata_new, extract=True,
                  disp=False)

    # Copy content of base_dir/directory/total to data.BASE_DIR/directory
    destination = join(data.BASE_DIR, directory)
    os.makedirs(destination, exist_ok=True)
    with os.scandir(join(base_dir, directory, 'total')) as it:
        for entry in it:
            if entry.is_file():
                shutil.copy(entry.path, destination)

    # Print some stats
    stop_time = perf_counter()
    print(f'\nCleanup completed in {stop_time - start_time:.1f} s.')
    print(f'A total of {len(metadata)} identifiers where handled and {count} '
          f'results saved.')


def _add_result_to_dict(content, content_dict):
    """
    Adds content to dict.

    :param content: Content to add.
    :param content_dict: Dict to add to.
    :return: Updated dict.
    """
    for id_, result in content:
        if id_ not in content_dict:
            content_dict[id_] = []
        if not _result_in_results(result, content_dict[id_]):
            content_dict[id_].append(result)
    return content_dict


def _result_in_results(result, results):
    """
    Checks if result is in results (OptResults objects can't be compared).

    :param result: result object to check for.
    :param list results: list of results to check in.
    :return:
    """
    return any(_cmp(result, x) for x in results)


def _cmp(x, y):
    """
    Used to compare objects containing (among other things) numpy arrays.

    TODO: better solution?

    :param x:
    :param y:
    :return:
    :rtype: bool
    """
    try:
        np.testing.assert_equal(x, y, verbose=False)
        return True
    except AssertionError:
        return False


def _cleanup_small(metadata):
    """
    Cleanup metadata.

    :param metadata: unclean metadata
    :return: cleaned metadata
    """
    meta_dict = {}
    for x in metadata:
        # Keep only the last exception for given identifier.
        if x[0] not in meta_dict or meta_dict[x[0]] is not True:
            meta_dict[x[0]] = x[1]
    return [[x, meta_dict[x]] for x in meta_dict]


def _get_metadata(directory, warn=True):
    """
    Look for metadata in directory. Returns [], {} if not found.

    :param directory: Directory to search through.
    :return: Metadata and meta-metadata.
    """
    try:
        return data.load(join(directory, 'total', 'metadata'),
                         base_dir=base_dir)
    except FileNotFoundError:
        if warn:
            warnings.warn('Metadata has not been initialized correctly, '
                          'run parallel.run with init=True and then with '
                          'cleanup=True.')
        return [], {}


def _mark_running(directory, task):
    """
    Mark as running.

    :param directory: Directory to save file in.
    :param task: Task (run, init, or cleanup)
    :return:
    """
    dir_complete = join(base_dir, directory)
    os.makedirs(dir_complete, exist_ok=True)
    Path(join(dir_complete, f'{task}_{mac}')).touch()


def _mark_not_running(directory, task):
    """
    Mark as not running (by removing the file created by _mark_running).

    :param directory: Directory to remove file from.
    :param task: Task (run, init, or cleanup)
    :return:
    """
    try:
        Path(join(base_dir, directory, f'{task}_{mac}')).unlink()
    except FileNotFoundError:
        pass


def _is_running(directory, tasks):
    """
    Check if anyone is running.

    :param directory: Directory to check in.
    :param tasks: tuple with tasks to check for.
    :return:
    """
    with os.scandir(join(base_dir, directory)) as it:
        for entry in it:
            if entry.is_file() and entry.name.startswith(tasks) \
                    and not entry.name.endswith(str(mac)):
                return True
    return False
