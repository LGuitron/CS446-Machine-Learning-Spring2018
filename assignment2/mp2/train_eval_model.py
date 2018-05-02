"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    if(shuffle):
        processed_dataset = shuffleData(processed_dataset)

    # Current dataset index
    currentIndex = 0
    for i in range(num_steps):
        # Epoch will end after this step
        # Only take remaining data rows for gradient descent
        if(currentIndex+batch_size > len(processed_dataset[1])):
            x = processed_dataset[0][currentIndex:len(processed_dataset[1])]
            y = processed_dataset[1][currentIndex:len(processed_dataset[1])]
            currentIndex = len(processed_dataset[1])

        else:
            x = processed_dataset[0][currentIndex:currentIndex+batch_size]
            y = processed_dataset[1][currentIndex:currentIndex+batch_size]
            currentIndex += batch_size

        # Reset index and shuffle if necesary
        if (currentIndex >= len(processed_dataset[1])):
            currentIndex = 0
            if(shuffle):
                processed_dataset = shuffleData(processed_dataset)

        update_step(x, y, model, learning_rate)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(x_batch)
    grad = model.backward(f, y_batch)
    model.w = model.w - learning_rate*grad


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    # Deep copy of dataset x values for adding 1 and computing the inverse
    # Without modifying the original dataset
    dataset_x_copy = np.copy(processed_dataset[0])
    ones = np.ones((len(dataset_x_copy), 1))
    dataset_x_copy = np.concatenate((dataset_x_copy, ones), axis=1)
    temp = np.matmul(np.transpose(dataset_x_copy), dataset_x_copy)
    temp += 2*model.w_decay_factor * np.identity(len(model.w))
    temp = np.linalg.pinv(temp)
    temp = np.matmul(temp, np.transpose(dataset_x_copy))
    model.w = np.matmul(temp, processed_dataset[1])
    return model


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    f = model.forward(processed_dataset[0])
    loss = model.total_loss(f, processed_dataset[1])
    return loss


# Shuffles dataset randomly
def shuffleData(processed_dataset):
    # Permutation for shuffling x and y numpy arrays together
    permutation = np.random.permutation(len(processed_dataset[1]))
    # Make temporal variable for previous dataset
    xOriginal = np.copy(processed_dataset[0])
    yOriginal = np.copy(processed_dataset[1])
    original_dataset = [xOriginal, yOriginal]

    # Shuffle both x and y numpy arrays with the same permutation
    for old_idx, new_idx in enumerate(permutation):
        processed_dataset[0][new_idx] = original_dataset[0][old_idx]
        processed_dataset[1][new_idx] = original_dataset[1][old_idx]

    return processed_dataset
