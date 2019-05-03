from multiprocessing import Pool
from functools import lru_cache


def identifier_generator():
    for x in range(5):
        yield x,


@lru_cache(maxsize=1)
def input_1(x):
    return 2*x,


input_functions = {1: input_1}


def simulate(x, y):
    return x**2


class Bookkeeper:
    """
    Class for keeping track of what's been done and only assign new tasks.
    """

    def __init__(self, iterator, book, output_calc=None):
        """

        :param iterator: Iterable
        :param set book: Set of identifiers corresponding to previously
            completed tasks.
        :param output_calc: List of functions
        """
        if output_calc is None:
            output_calc = {}
        self.iterator = iterator
        self.book = book
        self.output_calc = output_calc

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            x = self.iterator.__next__()
            if x not in self.book:
                output = []
                for i in range(len(x) + 1):
                    if i in self.output_calc:
                        output.extend(self.output_calc[i](
                            *[y for j, y in enumerate(x) if j < i]))
                return x, output


def wrap(x):
    return x[0], simulate(*x[0], *x[1])


ids = set()
generator = Bookkeeper(identifier_generator(), ids, input_functions)

with Pool(4, maxtasksperchild=1) as p:
    result_generator = p.imap_unordered(wrap, generator,
                                        chunksize=1)
    for id, res in result_generator:
        print(id)
        print(res)
        print('\n')
