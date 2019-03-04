import itertools
import numpy as np


def iterable(x):
    try:
        for _ in x:
            return True
    except TypeError:
        return False
    return True


# Warning: doesn't check for consistent depth, returns the first depth
# Warning: strings are subscriptable
def subscriptdepth(x):
    depth = 0
    while True:
        try:
            if type(x) is np.ndarray:
                return depth + 1
            x = x[0]
            depth += 1
            if type(x) is str:
                return depth
        except (TypeError, IndexError) as e:
            if type(e) is IndexError and str(e) == "list index out of range":
                depth += 1
            return depth


# Warning: might return bad things for numpy arrays (and other types)
def recursive_len(x):
    if subscriptdepth(x) and type(x) is not str:
        return sum(recursive_len(y) for y in x)
    elif type(x) is str:
        return len(x)
    elif type(x) is np.ndarray:
        return x.size
    else:
        return 1


def transposelist(x):
    temp = subscriptdepth(x)
    if temp == 0:
        return [[x]]
    elif temp == 1:
        x = [x]
    return list(map(list, itertools.zip_longest(*x)))


# Finds factor pair of x closest to sqrt(x)
# Typecasts x to int if necessery
# Returns [0, 0] for 0
def factor_sqrt(x):
    if type(x) is not int:
        x = int(x)
    if x == 0:
        return [0, 0]
    factor1 = None
    for i in range(int(np.sqrt(x)), 0, -1):
        if x % i == 0:
            factor1 = i
        if factor1 is not None:
            break
    factor2 = x // factor1
    if factor1 * factor2 == x:
        return [factor1, factor2]
    else:
        raise ValueError("There seems to be a numerical error, factor_sqrt was not successfuls")
