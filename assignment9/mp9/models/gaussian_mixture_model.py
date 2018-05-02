"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""
    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """
        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters (Done at the beginning of fit function)
        # np.array of size (n_components, n_dims)
        self._mu = np.zeros((self._n_components, self._n_dims))
        
        # Initialized with uniform distribution.
        # np.array of size (n_components, 1)
        self._pi = np.random.uniform(0,1,(self._n_components, 1)) 
        
        # Initialized with identity.
        # np.array of size (n_components, n_dims, n_dims)
        self._sigma = np.zeros((self._n_components, self._n_dims, self._n_dims))
        for i in range(self._n_components):
            self._sigma[i] = 1000*np.identity(self._n_dims)

    def fit(self, x):
        # Initialize means with points in x
        indices = np.random.choice(len(x), self._n_components, replace=False)
        for i in range(self._n_components):
            self._mu[i] = x[indices[i]]
            
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """
        for i in range(self._max_iter):
            posterior = self.get_posterior(x)
            self._m_step(x, posterior)
            print("i: " , i)

    def _e_step(self, x):
        
        
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        self._pi = np.sum(z_ik, axis=0)/len(x)
        self._mu = np.matmul(np.transpose(z_ik),x)
        self._mu = np.multiply(self._mu, (1/(len(x)*self._pi))[:,np.newaxis])
        
        for j in range(self._n_components):
            delta = x-self._mu[j,:]
            # Multiplication with z_ik and delta
            tmp = np.multiply((z_ik[:,j])[:, np.newaxis], delta)
            self._sigma[j] = np.matmul(np.transpose(tmp), delta)/(len(x)*self._pi[j]) 
            
            # Add reg_covar to the covariance matrices
            self._sigma[j] = self._sigma[j] + self._reg_covar*np.identity(self._n_dims)
            
    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N, n_components).
        """
        conditional = np.zeros((len(x),self._n_components))
        # Calculate each of the conditional probabilities according to normal distribution (pi_k N(xi | mu_k, sigma_k))
        for j in range(self._n_components):
            prob = self._multivariate_gaussian(x, self._mu[j], self._sigma[j])
            conditional[:, j] = prob
        
        return conditional

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        marginals = np.zeros(len(x))
        for i in range(self._n_components):
            marginals += self._pi[i] * self._multivariate_gaussian(x, self._mu[i], self._sigma[i])
        return marginals
        
    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        marginals = self.get_marginals(x)
        conditional = self.get_conditional(x)
        z_ik = (conditional * self._pi.reshape((-1))) / (marginals.reshape(-1,1) + self._reg_covar)
        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """
        self.cluster_label_map = []
        count_matrix = np.zeros((self._n_components, 10))
        max_indices = np.argmax(self.get_posterior(x), axis=1)
        for i in range(len(x)):
            count_matrix[max_indices[i], int(y[i])] += 1
        self.cluster_label_map = np.argmax(count_matrix, axis=1)
        self.cluster_label_map = list(self.cluster_label_map)
    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """
        y_hat = []
        max_indices = np.argmax(self.get_posterior(x), axis=1)
        for i in range(len(x)):
            y_hat.append(self.cluster_label_map[max_indices[i]])
        return np.array(y_hat)
