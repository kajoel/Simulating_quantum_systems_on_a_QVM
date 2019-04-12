from core import data
from constants import ROOT_DIR
from os.path import join
import matplotlib2tikz as tikz
import matplotlib.pyplot as plt
import numpy as np
from core import lipkin_quasi_spin

from matplotlib import rc

# rc('font',**{'family':'serif','serif':['Computer Modern']})
rc('text', usetex=True)


def tikz_save(title):
    tikz.save(title + ".tex",
              figurewidth='14cm',
              figureheight='8cm',
              textsize=11.0,
              tex_relative_path_to_data=None,
              externalize_tables=False,
              override_externals=False,
              strict=True,
              wrap=True,
              add_axis_environment=True,
              extra_axis_parameters=['font =\\footnotesize', 'scale only axis'],
              extra_tikzpicture_parameters=None,
              dpi=None,
              show_info=False,
              include_disclaimer=True,
              standalone=False,
              float_format="{:.15g}", )


###############################################################################
# Color maps

color_map_blue = np.array(
    [(161, 218, 180), (65, 182, 196), (34, 94, 168), (37, 52, 148)]) / 255

color_map_red = np.array([(254, 240, 217), (253, 204, 138), (252, 141, 89),
                          (215, 48, 31)]) / 255

color_map_gray = np.array(
    [(204, 204, 204), (150, 150, 150), (99, 99, 99), (37, 37, 37)]) / 255

###############################################################################
# Line styles
linestyles = ['-', '--', '-.', ':']
linewidth = 1
labels = [r'one\_particle', r'one\_particle\_ucc', r'multi\_particle',
          'multi\_particle\_ucc']
fontsize = 10
###############################################################################
# PLOT
###############################################################################
datatitle = 'ansatz_depth'
datatitle = join(ROOT_DIR, 'data/mat_to_op and ansatz', datatitle + '.pkl')

data_, metadata = data.load(datatitle)
print(metadata, '\n')

num_of_plot = 4

for i in range(num_of_plot):
    plt.plot(data_['mat_size'][i], data_['gates'][i], color=color_map_gray[i],
             marker='o', linewidth=linewidth, markersize=2,
             label=labels[i])

plt.xlabel(r'Matrisstorlek', fontsize=fontsize)
plt.ylabel(r'Antal kvantgrindar', fontsize=fontsize)
plt.ylim(0, 10000)
plt.xlim(0, 100)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

x = [data_['mat_size'][i][-1] for i in range(4)]
y = [data_['gates'][i][-1] for i in range(4)]
plt.text(x[0] + 2, 9000, labels[0])
plt.text(x[1] - 2, 8000, labels[1])
plt.text(x[2] - 20, y[2] + 300, labels[2], fontsize=fontsize)
plt.text(x[3], 9000, labels[3], fontsize=fontsize,
         bbox=dict(facecolor='white', edgecolor='white', pad=1))
plt.show()

#tikz_save('samples_gates')

'''
data_ = {'Samples' : samples,'Expected_values': exp_value,
             'Parameter_error': para_error,'Variance': variance, 
             'Iterations': iterations}


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
'''
