from core import data
from constants import ROOT_DIR
from os.path import join
import matplotlib2tikz as tikz
import matplotlib.pyplot as plt
import numpy as np
from core import lipkin_quasi_spin

def create(title):
    tikz.save(title + ".tex")


###############################################################################
# TEST
###############################################################################
datatitle = join(ROOT_DIR, 'data', 'SampleErrorRunj2i1V1.pkl')

data_, metadata = data.load(datatitle)
print(data_['Expected_values'])
print(metadata.keys())

'''
data_ = {'Samples' : samples,'Expected_values': exp_value,
             'Parameter_error': para_error,'Variance': variance, 
             'Iterations': iterations}
'''

facit = lipkin_quasi_spin.eigs(2,1)[1]
print(facit)
facit = float(facit[0])

samples = data_['Samples']
exp_val = data_['Expected_values']
variance = data_['Variance']
start = samples[0]
stop = samples[-1]


fig = plt.figure()  # create a figure object
ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure
#ax.plot([1, 2, 3, 4])
#ax.set_ylabel('some numbers')


plot_ = ax.hlines(facit, start, stop, colors='r', linestyles='dashed',
           label='True eig: {}'.format(round(facit, 4)))
ax.errorbar(samples, exp_val, variance, fmt='o',
             label='Eig calculated with Nelder-Mead')
ax.legend()
ax.set_xlabel('Samples')
ax.set_ylabel('Eigenvalue')

plt.show()
#create('test')
