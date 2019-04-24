''''
Parameters in the saved data dict
'para_error': [],
'variance': [],
'n_calls': [],
'samples': [],
'result': []
'''

from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join

#name of the files without the case at the end. Case is from 0 to 10
file_name = 'bayes_parallel_one_particle_ucc_updated_0'
file_path = join('bayes_total_evals', file_name)

data, _ = data.load(file_path)
for key in data:
    print(key)
    print(data[key])
