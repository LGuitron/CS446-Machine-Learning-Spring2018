"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model."""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,1).
        """
        # MSE derivative
        dL = np.zeros((len(self.x[0]), 1))
        dL[:, 0] = (1/np.size(f, 0))*np.matmul(np.transpose(f)-y, self.x)
        # L2 regularization derivative
        dR = (1/np.size(f, 0))*self.w_decay_factor*self.w
        total_grad = dL+dR
        return total_grad

    def total_loss(self, f, y):
        """Computes the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """
        squareLoss = (1/(2*np.size(f, 0)))*np.sum((y-np.transpose(f))**2)
        regLoss = (1/(2*np.size(f, 0)))*self.w_decay_factor * np.sum(self.w**2)
        total_loss = squareLoss + regLoss
        return total_loss

    def predict(self, f):
        """Nothing to do here.
        """
        return f
