
from core import data
import numpy as np
import matplotlib.pyplot as plt
from constants import ROOT_DIR
from os.path import join

version = 3
directory = f'bayes_total_evals/v{version}'


def load_data(ansatz_, size, matidx):
    file_name = f'{ansatz_}_size={size}_matidx={matidx}'

    file_path = join(directory, file_name)
    return data.load(file_path)


def load_multiple(ansatz_, size, mean=False):
    if mean is True: partition = mean_partitan_data
    else: partition = partitan_data

    temp_dict = {}
    for index in range(4):
        try:
            temp_dict[index] = partition(load_data(ansatz_, size, index))
        except:
            print(f"An exception occurred with matrix of index: {index}")
    return temp_dict
  
def partitan_data(data_dict):

    number_of_datapoints = len(data_dict[0])

    data_matrix = np.zeros( (6, number_of_datapoints) )
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
    # Select data do not work right now  
    data_matrix = select_data(data_matrix)
    return data_matrix, parameter


def select_data(data_matrix, samples_low=100, n_calls_low=5, repeats=5):
    uniq_samples = np.unique(data_matrix[2])
    uniq_calls = np.unique(data_matrix[3])

    for index_calls in range(len(uniq_calls)):
        if uniq_calls[index_calls]>=n_calls_low: break

    for index_samples in range(len(uniq_samples)):
        if uniq_samples[index_samples]>=samples_low: break
    

    uniq_samples = uniq_samples[index_samples:]
    uniq_calls = uniq_calls[index_calls:]
  

    number_of_datapoints = len(uniq_calls)*len(uniq_samples)
        
    new_matrix = np.zeros( (6, number_of_datapoints) )
    
    
    j = 0
    count = 0
    temp = np.zeros(repeats)
    
    for i in range(len(data_matrix[0])):
        if data_matrix[3,i] < n_calls_low or data_matrix[2,i] < samples_low:
            continue
        if j == 0:        
            # Samples
            new_matrix[2,count] = data_matrix[2,i]
            # n_calls
            new_matrix[3,count] = data_matrix[3,i]
            # correct value
            new_matrix[5,count] = data_matrix[5,i]
            # Evaluations on the quantum computer
            new_matrix[0,count] = data_matrix[0,i]
        
        temp[j] = data_matrix[4,i]
        j+=1
    
        if j ==repeats:
            # eigenvalue
            new_matrix[4,count] = np.mean(temp)
            # error in eigenvalue
            new_matrix[1,count] = np.abs( np.linalg.norm(new_matrix[4,count] - 
                                                 new_matrix[5,count])
                                                 / new_matrix[5,count] ) * 100
            j = 0
            count +=1
        

    return new_matrix
        
    

def mean_partitan_data(data_dict, start_n_calls = 15, repeats=5):
    for start_index, data_point in enumerate(data_dict[0]):
        if data_point[0][4] >= start_n_calls: break

    number_of_datapoints = len(data_dict[0]) - start_index  

    data_matrix = np.zeros( (6, int(number_of_datapoints/5)) )

    parameter = []
    temp = np.zeros(5)
    i, count = 0, 0
    for index,data_point in enumerate(data_dict[0]):
        if index < start_index: continue
        if i == 0:        
            # Samples
            data_matrix[2,count] = data_point[0][3]
            # n_calls
            data_matrix[3,count] = data_point[0][4]
            # correct value
            data_matrix[5,count] = data_point[1]['correct']
            # Evaluations on the quantum computer
            data_matrix[0,count] = data_matrix[2,count]*data_matrix[3,count]
        
        
        temp[i] = data_point[1]['fun']
        i+=1
        if i ==repeats:
            # eigenvalue
            data_matrix[4,count] = np.mean(temp)
            # error in eigenvalue
            data_matrix[1,count] = np.linalg.norm(data_matrix[5,count] - 
                                                  data_matrix[4,count])
            i = 0
            count +=1

    return data_matrix, parameter


def heatmap(ansatz_, size):
    from pandas import DataFrame
    import seaborn as sns
    from scipy.interpolate import griddata

    temp_dict = load_multiple(ansatz_, size, mean=False)
    plt.figure()
    list_of_matrices = []
    for i, value in temp_dict.items():
        
        samples = np.unique(value[0][2])
        n_calls = np.unique(value[0][3])
        eig_error = value[0][1]
        

        '''
        x = value[0][2]
        y = value[0][3]
        #y = [x[i]*temp_y[i] for i in range(len(x))]

        z = value[0][1]
        
        xi = np.linspace(np.unique(x)[0], np.unique(x)[-1], int(len(x)/2))
        yi = np.linspace(np.unique(y)[0], np.unique(y)[-1], int(len(y)/2))
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
        plt.figure()
        CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
        CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
        plt.colorbar() # draw colorbar
        # plot data points.
        plt.scatter(x,y,marker='o',c='b',s=5)

        '''

        error_mesh = np.zeros( (len(samples), len(n_calls)) )

        
        for i_1 in range(len(n_calls)):
            for i_2 in range(len(samples)):
                try:
                    error_mesh[i_2,i_1] = eig_error[i_1*len(samples)+i_2]
                except:
                    print(f"Ofullständig data för heatmap: {i}")
        list_of_matrices.append(error_mesh)
    final_matrix = np.zeros( list_of_matrices[0].shape )
    for value in list_of_matrices:
        final_matrix += value * 0.25
    
    for i, row in enumerate(final_matrix):
        print('\n')
        for j, value in enumerate(row):
            print(f'{n_calls[j]*samples[i]}\t{samples[i]}\t{value}')

    
    df = DataFrame(final_matrix, index=samples, columns=n_calls)  
    sns.heatmap(df, vmax=20)       

        
        



def plot_all_of_size(ansatz_, size, mean = False):
    temp_dict = load_multiple(ansatz_, size, mean=mean)
    
    if ansatz_ == 'multi_particle':c = 'r'
    else: c = 'k'

    for i, value in temp_dict.items():
        if i == 0: label = ansatz_
        else: label = None
        plt.scatter(value[0][0], value[0][1], s =10**2 , c=c, label=label, )
    plt.xlabel('Evaluations on quantum computer')
    plt.ylabel('Eigenvalue error')
    plt.legend()

################################################################################
# Main
################################################################################
if __name__ == '__main__':
    size = 3

    plt.figure()
    plot_all_of_size('multi_particle', size)
    plot_all_of_size('one_particle_ucc', size)
    
    
    heatmap('multi_particle', size)    
    heatmap('one_particle_ucc', size)  
    
    plt.show()




