"""Input and output helpers to load in data.
"""
import numpy as np


def read_dataset(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of each sample

        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...]
                             where yi is +1/-1, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')

    A = []
    T = []

    indexFile = open(path_to_dataset_folder+'/'+index_filename, 'r')
    for sample in indexFile:
        last_y = [-1]
        if(sample[0] == '1'):
            last_y = [1]

        # Open sample file
        sampleLoc = sample.split('/')
        sampleLoc = sampleLoc[1].split('\n')
        sampleF = open(path_to_dataset_folder+'/samples/'+sampleLoc[0], 'r')

        # Get feature vectors from files
        for line in sampleF:
            c = 0
            last_x = [1]
            line = line.split(' ')
            for i in range(len(line)-1):
                if(line[i] != ''):
                    last_x.append(float(line[i]))

            # Remove \n for last element
            temp = line[len(line)-1].split('\n')
            last_x.append(float(temp[0]))

        A.append(last_x)
        T.append(last_y)

    A = np.array(A)
    T = np.array(T)

    return A, T
