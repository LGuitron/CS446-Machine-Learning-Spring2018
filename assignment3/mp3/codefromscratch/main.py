"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 1E-6
max_iters = 100000

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset','indexing.txt')
    
    # Initialize model.
    ndim = A.shape[1]
    model = LogisticModel(ndim, 'zeros')
    
    # Train model via gradient descent.
    model.fit(T, A, learn_rate, max_iters)

    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')
    
    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')
    #print(model.W)

    # Try all other methods: forward, backward, classify, compute accuracy
    pred = model.classify(A)
    accuracy = 1 - 0.5*np.sum(T[:,0] - pred)/len(pred)
    print(model.W)
    print("Accuracy: " , accuracy)
    pass 

