# Carl 4/5
###############################################################################
# Imports
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join
import itertools
from plot import tikzfigure

def plot_gates():
    base_dir = join(ROOT_DIR, f'data/mat_to_op and ansatz')
    file = f'ansatz_depth.pkl'

    data_, metadata = data.load(file, base_dir)
    gates = data_['gates']
    mat_size = data_['mat_size']


    for i in range(4):
        plt.plot(mat_size[i],gates[i], '-o')

def plot_numterms():
    base_dir = join(ROOT_DIR, f'data/mat_to_op and ansatz')
    file = f'num_terms.pkl'
    data2, metadata = data.load(file, base_dir)
    print(metadata)
    for i in range(2):
        plt.plot(data2['mat_size'][i], data2['gates'][i])


plot_gates()
tikzfigure.save('gates')
plt.show()