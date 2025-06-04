import numpy as np
import networkx as nx
from gnar.neighbours import get_neighbours
from gnar import matrices
from scipy.sparse.linalg import spsolve

class GNARFit:
    """
        Fit the Generalized Network Autoregressive (GNAR) model to time series data.
        -----------------------------------
        USAGE:
        1. Import the class:
            from gnar.models.gnarFit import GNARFit
        2. Create an instance of the class:
            gnar_fit = GNARFit(net, alphaOrder, betaOrder, data, globalAlpha=False, globalBeta=False)
        3. Fit the model:
            alpha, beta = gnar_fit.fit()
        
            
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
            alpha = np.zeros((self.K, self.alphaOrder))
        else:
            alpha = np.zeros(self.alphaOrder)
        if self.globalBeta:
            beta = np.zeros((self.K, self.K, self.betaOrder))
        else:
            beta = np.zeros((self.K, self.betaOrder))
        
        Z = matrices.Zmatrix(self.data, self.alphaOrder)
        R, index_map, gamma_index_map = matrices.Rmatrix(self.net, self.alphaOrder, self.betaOrder, global_alpha=self.globalAlpha, global_beta=self.globalBeta )
        y = matrices.yvector(self.data)
        A = np.kron(Z.T, np.eye(self.K)) @ R
        gamma = spsolve(A,y)

        # Fill in alpha and beta
        for i in range(self.K):
            for j in range(self.K):
                for k in range(self.betaOrder):
                    if self.globalBeta:
                        beta[i,j,k] = gamma[gamma_index_map[(i,j,k)]]
                    else:
                        beta[i,k] = gamma[gamma_index_map[(i,k)]]
            for k in range(self.alphaOrder):
                if self.globalAlpha:
                    alpha[i,k] = gamma[index_map[(i,k)]]
                else:
                    alpha[k] = gamma[index_map[k]]
        
        return alpha, beta
    
    
    
