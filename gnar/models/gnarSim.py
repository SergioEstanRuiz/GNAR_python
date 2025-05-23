import numpy as np
import networkx as nx
from gnar.neighbours import get_neighbours

class GNARSimulator:
    """
        Simulator for the Generalized Network Autoregressive (GNAR) model.
        -----------------------------------
        USAGE:
        1. Import the class:
            from gnar.models.gnar import GNARSimulator
        2. Create an instance of the GNARSimulator class with the desired parameters:
            gnar = GNARSimulator(net, alpha, beta, sigma)
        3. Generate synthetic data using the generate_gnar method:
            X = gnar.generate_gnar(T, burnin, seed)
        -----------------------------------
    """

    def __init__(self, net, alpha, beta, sigma=1.0):
        """
        Initialize the GNAR model.

        Parameters:
        -----------
        net : np.ndarray
            Adjacency matrix (N x N).
        alpha : np.ndarray
            Matrix of autoregressive coefficients (N x P).
            E.g., alpha[0][1] = alpha for node=1, lag = 1
        beta : list[np.ndarray]
            List determines the lag, then we have a (N x s_j) matrix for each lag j.
            E.g., beta[2][0][1] = beta for lag=3, node=1, neighbour=2

        sigma : float
            Standard deviation of Gaussian noise.
        """
        self.net = net
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

    def generate_gnar(self,T, burnin=50, seed=None):
        """
        Generate synthetic data from a GNAR model using disjoint neighbor sets.

        Parameters:
        -----------
        T : int
            Number of time points (after burn-in).
        burnin : int
            Initial time points to discard.
        seed : int or None
            Random seed.

        Returns:
        --------
        X : np.ndarray
            Time series data (T x N).
        """
        if seed is not None:
            np.random.seed(seed)

        G = nx.from_numpy_array(self.net)
        N = self.net.shape[0]
        lag = len(self.alpha[0])
        max_stage = max(len(b[0]) for b in self.beta)
        total_T = T + burnin

        # Get neighbour sets
        neighbor_sets = get_neighbours(self.net, max_stage)
        # Initialize time series
        X = np.zeros((total_T, N))

        for t in range(lag, total_T):
            X_t = np.zeros(N)
            for p in range(1, lag + 1):
                X_t += self.alpha[p - 1] * X[t - p]
                for j in range(1, len(self.beta[p - 1]) + 1):
                    for i in range(N):
                        neighs = neighbor_sets[i].get(j, [])
                        if neighs:
                            avg = np.mean(X[t - p, neighs])
                            X_t[i] += self.beta[p - 1][j - 1] * avg
            X_t += np.random.normal(scale=self.sigma, size=N)
            X[t] = X_t

        return X[burnin:]

