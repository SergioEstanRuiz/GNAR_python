from gnar.models.gnarSim import GNARSim
import numpy as np
from gnar.models.gnarFit import GNARFit
import networkx as nx

K = 5
net = nx.to_numpy_array(nx.erdos_renyi_graph(K, 0.5))
# alpha = 0.2*np.random.randn(2)  
# beta = [0.1*np.random.randn(2) for _ in range(2)]
alpha = np.array([[0.5],[0],[0],[-0.5],[0]]) 
beta = [np.array([0])]
print(f"Alpha coefficients: {alpha}")
print(f"Beta coefficients: {beta}")
GNARSim = GNARSim(net, alpha, beta, sigma=0.1)
data = GNARSim.generate_gnar(T=1000, X_start=None, burnin=50, seed=42)

gnarFit = GNARFit(net, alphaOrder=1, betaOrder=[1], data=data[:,:-1], globalAlpha=False, globalBeta=True)
print(f"Fitted Alpha coefficients: {gnarFit.alpha}")
print(f"Fitted Beta coefficients: {gnarFit.beta}")

predictions = gnarFit.predict(h=1)
print(f"Predictions for next time step: {predictions}")
print(f"Actyal next time step: {data[:, -1]}")

