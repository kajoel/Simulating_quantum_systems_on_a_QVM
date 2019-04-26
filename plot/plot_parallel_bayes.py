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

version = 3
directory = f'bayes_total_evals_v{version}'


def load_data(ansatz_, size, matidx):
    file_name = f'{ansatz_}_size={size}_matidx={matidx}'
    if ansatz_ == 'multi_particle': temp_direc = directory + '/247163800758493'
    else: temp_direc = directory + '/176156460388480'

    file_path = join(temp_direc, file_name)
    return data.load(file_path)


def load_multiple(ansatz_, size):
    temp_dict = {}
    for index in range(4):
        try:
            temp_dict[index] = partitan_data(load_data(ansatz_, size, index))
        except:
            print(f"An exception occurred, missing matrif of index: {index}")
    return temp_dict
  

def partitan_data(data_dict):
    data_matrix = np.zeros( (6, len(data_dict[0])) )
    parameter = []
    for index, data_point in enumerate(data_dict[0]):
        # Samples
        data_matrix[2,index] = data_point[0][3]
        # n_calls
        data_matrix[3,index] = data_point[0][4]
        # eigenvalue
        data_matrix[4,index] = data_point[1]['fun']
        # correct value
        data_matrix[5,index] = data_point[1]['correct']
        # error in eigenvalue
        data_matrix[1,index] = np.linalg.norm(data_matrix[5,index] - 
                                              data_matrix[4,index])
        # Evaluations on the quantum computer
        data_matrix[0,index] = data_matrix[2,index]*data_matrix[3,index]
        
        # parameter
        parameter.append(data_point[1]['x'])
    return data_matrix, parameter



def plot_all_of_size(ansatz_, size):
    temp_dict = load_multiple(ansatz_, size)

    for _, value in temp_dict.items():
        plt.scatter(value[0][0], value[0][1])


################################################################################
# Main
################################################################################
if __name__ == '__main__':
    plt.figure()
    plot_all_of_size('multi_particle', 2)
    plt.figure()
    plot_all_of_size('one_particle_ucc', 2)
    plt.show()




