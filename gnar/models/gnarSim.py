import numpy as np
import networkx as nx
from gnar.neighbours import get_neighbours

class GNARSim:
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
            Adjacency matrix (K x K).
        alpha : np.ndarray
            Matrix of autoregressive coefficients (K x P) if alpha is local
            E.g., alpha[0][1] = alpha for node=1, lag = 2 
            OR
            Matrix of autoregressive coefficients (P) if alpha is global
            E.g., alpha[0] = alpha for lag = 1 if alpha is global.
        beta : list[np.ndarray]
            List determines the lag, then we have a (K x s_j) matrix for each lag j if beta is local,
            E.g., beta[2][0][1] = beta for lag=3, node=1, neighbour=2, 
            or beta[2][0] = beta for lag=3, neighbour=2 if beta is global.

        sigma : float
            Standard deviation of Gaussian noise.
        """
        self.net = net
        self.alpha = alpha
        if alpha.ndim == 1:
            self.lag = len(alpha)
        else:
            self.lag = alpha.shape[1]
        if beta[0].ndim == 1:
            self.maxStage = max(len(b) for b in beta)
        else:
            self.maxStage = max(len(b[0]) for b in beta)
        self.beta = beta
        self.sigma = sigma
        self.globalAlpha = True if alpha.ndim ==1 else False
        self.globalBeta = True if beta[0].ndim == 1 else False
        self.K = net.shape[0]  # Number of nodes

    def generate_gnar(self,T, X_start = None, burnin=50, seed=None):
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
        X_start : np.ndarray
            Initial values for the time series (K x lag).

        Returns:
        --------
        X : np.ndarray
            Time series data (K x T).
        """
        if seed is not None:
            np.random.seed(seed)

        G = nx.from_numpy_array(self.net)
        total_T = T + burnin

        # Get neighbour sets
        neighbor_sets = get_neighbours(self.net, self.maxStage)
        # Initialize time series
        X = np.zeros((self.K, total_T))

        if X_start is not None:
            ncols = min(self.lag, X_start.shape[1])
            X[:, :ncols] = X_start[:, -ncols:]            

        for t in range(self.lag, total_T):
            X_t = np.zeros(self.K)  # Adjusted to self.K (number of nodes)
            for p in range(1, self.lag + 1): # Goes from 1 to self.lag
                # Handle global/local alpha
                if self.globalAlpha:
                    X_t += self.alpha[p - 1] * X[:, t - p]
                else:
                    X_t += self.alpha[:, p - 1] * X[:, t - p]

                # Handle global/local beta
                for j in range(1, len(self.beta[p - 1]) + 1):
                    for i in range(self.K):
                        # Get neighbors for node i at distance j, returns array of nodes
                        neighs = neighbor_sets[i].get(j, [])
                        if neighs:
                            avg = np.mean(X[neighs, t - p])  # Adjusted to use neighbors' rows
                            if self.globalBeta:
                                X_t[i] += self.beta[p - 1][j - 1] * avg
                            else:
                                X_t[i] += self.beta[p - 1][i, j - 1] * avg

            # Add Gaussian noise
            X_t += np.random.normal(scale=self.sigma, size=self.K)
            X[:, t] = X_t  # Update the column for time `t`

        return X[:, burnin:]

