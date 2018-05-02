"""Input and output helpers to load in data.
"""

import pickle
import numpy as np
from numpy import genfromtxt


def read_dataset(input_file_path):
    """Read input file in csv format from file.
    In this csv, each row is an example, stored in the following format.
    label, pixel1, pixel2, pixel3...

    Args:
        input_file_path(str): Path to the csv file.
    Returns:
        (1) label (np.ndarray): Array of dimension (N,) containing the label.
        (2) feature (np.ndarray): Array of dimension (N, ndims) containing the
        images.
    """
    # Imeplemntation here.
    features = []
    labels = []

    indexFile = open(input_file_path, 'r')
    for sample in indexFile:
        sample = sample.split(',')
        label= float(sample[0])
        labels.append(label)
        
        sample_features = []
        for i in range(1,len(sample),1):
            feature = float(sample[i])
            sample_features.append(feature)
        features.append(sample_features)
        
    features = np.array(features)
    labels = np.array(labels)

    return labels, features
