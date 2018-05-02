"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 1E-4
max_iters = 1000


def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')

    # Initialize model.
    ndim = A.shape[1]
    model = LogisticModel_TF(ndim, 'zeros')

    # Build TensorFlow training graph
    model.build_graph(learn_rate)

    # Train model via gradient descent.
    session = tf.Session()
    sigmoid_res = model.fit(T, A, max_iters)

    # Compute classification accuracy based on the return of the "fit" method
    accuracy = session.run(model.calculate_accuracy, feed_dict={model.prediction: sigmoid_res, model.Y: T})
    print("Final Accuracy: ", accuracy)


if __name__ == '__main__':
    tf.app.run()
