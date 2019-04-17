"""
Run in parallel and save appropriately.

@author = Joel
"""
from core.data import save
from multiprocessing import Pool


def run(simulate, iterate, data, metadata, path, num_sim=5):
    """


    :param simulate: Function that takes input from iterate, runs simulation
        and modifies the data (and perhaps metadata) object.
    :param iterate: Iterable that yield tuples that will be unpacked an
        passed to simulate.
    :param data: the data object (preferably a dictionary)
    :param metadata: the metadata dictionary.
    :param path: Path to the file to save to.
    :param
    :return:
    """
    # Wrapper for simulate (iterate should yield tuples)
    def wrap(x):
        simulate(*x)

    # Empty iterator with num entries
    def runs(num, x):
        for i in range(num):
            yield x

    count = 0  # keep track of what's been done
    with Pool(num_sim, maxtasksperchild=1) as p:
        for x in iterate():
            count += 1
            if count > metadata['count']:
                # In this case the relevant simulation has not already been
                # saved to the file and is therefore run now.

                # Run a few times in parallel using multiprocess.Pool and map.
                # Assumes that simulate modifies the data and metadata objects.
                p.map(wrap, runs(num_sim, x))

                # Update metadata['count'] and save file:
                metadata['count'] = count
                save(path, data, metadata)
