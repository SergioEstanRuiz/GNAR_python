import torch
import torch.nn as nn
from gnar.neighbours import get_neighbours

class GNARLayer(nn.Module):
    def __init__(self, K, p, s_list, globalAlpha = True, globalBeta = True):
        super().__init__()
        self.K = K
        self.p = p
        self.s_list = s_list  # list of ints, e.g. [1, 2, 1]
        self.globalAlpha = globalAlpha
        self.globalBeta = globalBeta
        
        # Alpha parameters
        if globalAlpha:
            self.alpha = nn.Parameter(torch.randn(p))
        else:
            self.alpha = nn.Parameter(torch.randn(K, p))
        # Beta parameters
        if globalBeta:
            self.beta = nn.ParameterList([
                nn.Parameter(torch.randn(s)) for s in s_list
            ])
        else:
            self.beta = nn.ParameterList([
                nn.Parameter(torch.randn(K, s)) for s in s_list
            ])
        
    
    def forward(self, X, A):
        """
        Forward pass for the GNAR layer.

        Parameters:
        -----------
        X : torch.Tensor
            Input time series data of shape (K, T), where K is the number of nodes
            and T is the number of time steps.
        A : torch.Tensor
            Adjacency matrix of shape (K, K).

        Returns:
        --------
        torch.Tensor
            Output time series data of shape (K, T - p).
        """
        K, T = X.shape
        assert T > self.p, "Input time series must have more time steps than the number of lags (p)."

        # Initialize output tensor
        Y = torch.zeros(K, T - self.p, device=X.device)

        # Compute neighbor sets based on adjacency matrix
        neighbor_sets = get_neighbours(A.cpu().numpy(), max(self.s_list))

        # Iterate over time steps (starting from lag `p`)
        for t in range(self.p, T):
            Y_t = torch.zeros(K, device=X.device)

            # Apply alpha parameters (autoregressive terms)
            for lag in range(1, self.p + 1):
                if self.globalAlpha:
                    Y_t += self.alpha[lag - 1] * X[:, t - lag]
                else:
                    Y_t += self.alpha[:, lag - 1] * X[:, t - lag]

            # Apply beta parameters (neighbor terms)
            for lag in range(1, self.p + 1):
                for stage, s in enumerate(self.s_list, start=1):
                    for i in range(K):
                        neighbors = neighbor_sets[i].get(stage, [])
                        if neighbors:
                            avg_neighbors = X[neighbors, t - lag].mean()
                            if self.globalBeta:
                                Y_t[i] += self.beta[stage - 1][lag - 1] * avg_neighbors
                            else:
                                Y_t[i] += self.beta[stage - 1][i, lag - 1] * avg_neighbors

            Y[:, t - self.p] = Y_t

        return Y