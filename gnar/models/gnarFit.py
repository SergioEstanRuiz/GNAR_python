import numpy as np
import networkx as nx
from gnar.neighbours import get_neighbours
from gnar import matrices
from scipy.sparse.linalg import lsqr
from gnar.models.gnarSim import GNARSim

class GNARFit:
    """
        Fit the Generalized Network Autoregressive (GNAR) model to time series data.
        -----------------------------------
        USAGE:
        1. Import the class:
            from gnar.models.gnarFit import GNARFit
        2. Create an instance of the class:
            gnar_fit = GNARFit(net, alphaOrder, betaOrder, data, globalAlpha=False, globalBeta=False)
            Notice that this already fits the model and saves the parameters in self.alpha and self.beta.
        3. Make predictions using the predict method:
            predictions = gnar_fit.predict(h)
        
            
    """

    def __init__(self, net, alphaOrder, betaOrder, data, globalAlpha = False, globalBeta = False):
        self.net = net
        self.alphaOrder = alphaOrder
        self.betaOrder = betaOrder
        self.data = data
        self.globalAlpha = globalAlpha
        self.globalBeta = globalBeta
        self.K = net.shape[0]
        self.T = data.shape[0]
        self.alpha, self.beta = self.fit()

    def fit(self):
        if self.globalAlpha:
            alpha = np.zeros(self.alphaOrder)
        else:
            alpha = np.zeros((self.K, self.alphaOrder))
        if self.globalBeta:
            beta = [np.zeros(self.betaOrder[i]) for i in range(self.alphaOrder)]
        else:
            beta = [np.zeros((self.K, self.betaOrder[i])) for i in range(self.alphaOrder)]
        
        Z = matrices.Zmatrix(self.data, self.alphaOrder)
        R, index_map, gamma_index_map = matrices.Rmatrix(self.net, self.alphaOrder, self.betaOrder, global_alpha=self.globalAlpha, global_beta=self.globalBeta )
        y = matrices.yvector(self.data, p=self.alphaOrder)
        A = np.kron(Z.T, np.eye(self.K)) @ R
        gamma = lsqr(A,y)[0]

        # Fill alpha
        if self.globalAlpha:
            for k in range(self.alphaOrder):
                alpha[k] = gamma[gamma_index_map[('alpha', k+1)]]
        else:
            for i in range(self.K):
                for k in range(self.alphaOrder):
                    alpha[i, k] = gamma[gamma_index_map[('alpha', i, k+1)]]

        # Fill beta
        if self.globalBeta:
            for k in range(self.alphaOrder):
                for dist in range(1, self.betaOrder[k]+1):
                    beta[k][dist-1] = gamma[gamma_index_map[('beta', k+1, dist)]]
        else:
            for k in range(self.alphaOrder):
                for i in range(self.K):
                    for dist in range(1, self.betaOrder[k]+1):
                        beta[k][i, dist-1] = gamma[gamma_index_map[('beta', i, k+1, dist)]]
        
        return alpha, beta
    
    def predict(self, h):
        """
        Predict future values using the fitted GNAR model.

        Parameters:
        -----------
        h : int
            Number of steps ahead to predict.

        Returns:
        --------
        predictions : np.ndarray
            Predicted values (K x h).
        """
        
        predictor = GNARSim(self.net, self.alpha, self.beta, sigma =0)
        predictions = predictor.generate_gnar(T=h+self.alphaOrder, X_start=self.data[:,-self.alphaOrder:], burnin=0)
        predictions = predictions[:, -h:]
        return predictions
    
    
