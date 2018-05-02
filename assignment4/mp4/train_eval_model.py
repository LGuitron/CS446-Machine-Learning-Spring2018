"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def train_model(data, model, learning_rate=0.001, batch_size=32,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    if(shuffle):
        data = shuffleData(data)

    # Current dataset index
    currentIndex = 0
    for i in range(num_steps):
        # Epoch will end after this step
        # Only take remaining data rows for gradient descent
        if(currentIndex+batch_size > len(data['label'])):
            x = data['image'][currentIndex:len(data['label'])]
            y = data['label'][currentIndex:len(data['label'])]
            currentIndex = len(data['label'])
        else:
            x = data['image'][currentIndex:currentIndex+batch_size]
            y = data['label'][currentIndex:currentIndex+batch_size]
            currentIndex += batch_size

        # Reset index and shuffle if necesary
        if (currentIndex >= len(data['label'])):
            currentIndex = 0
            if(shuffle):
                data = shuffleData(data)

        update_step(x, y, model, learning_rate)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    f = model.forward(x_batch)
    grad = model.backward(f, y_batch)
    model.w = model.w - learning_rate*grad


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)
    # Set model.w
    model.w[:, 0] = z[0:len(model.w), 0]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    # z = (w, b, xi)
    # Points
    m = data['image'].shape[0]

    # Features
    f = data['image'].shape[1]
    P = np.zeros((m+f+1, m+f+1))

    # Set ones for each value corresponding to w
    for i in range(f):
        P[i][i] = 1

    q = np.ones((m+f+1, 1))
    # Set zeros for w and b
    for i in range(f+1):
        q[i] = 0

    # Set 1/wDecay for all xi
    q = (1/model.w_decay_factor) * q

    # Diagonal Matrix for y labels
    G = np.identity(len(data['label']))
    for i in range(len(data['label'])):
        G[i][i] = G[i][i]*data['label'][i]

    # Multiply matrix by image data
    G = np.matmul(G, data['image'])
    col = np.ones((m, 1))
    col[:, 0] = data['label']
    G = np.append(G, col, 1)
    G = np.append(G, np.identity(m), 1)
    G = -1*G
    h = -1*np.ones((m, 1))

    # Add lower bound restrictions to G and h
    lb_vector = np.zeros((m+f+1, 1))
    lb_matrix = -1*np.identity(m+f+1)
    
    # Append diagonal matrix with 1s for all xis, 
    # and 0 elsewhere (for lower bound restrictions)
    for i in range(f + 1):
        lb_matrix[i][i] = 0
    
    G = np.append(G, lb_matrix, 0)
    h = np.append(h, lb_vector, 0)

    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    f = model.forward(data['image'])
    loss = model.total_loss(f, data['label'])
    y_pred = model.predict(f)
    accVector = y_pred[:, 0] == data['label'][:]
    acc = np.sum(accVector)/len(accVector)
    return loss, acc


# Shuffles dataset randomly
def shuffleData(data):
    # Permutation for shuffling x and y numpy arrays together
    permutation = np.random.permutation(len(data['image']))
    # Make temporal variable for previous dataset
    xOriginal = np.copy(data['image'])
    yOriginal = np.copy(data['label'])

    # Shuffle both x and y numpy arrays with the same permutation
    for old_idx, new_idx in enumerate(permutation):
        data['image'][new_idx] = xOriginal[old_idx]
        data['label'][new_idx] = yOriginal[old_idx]

    return data
