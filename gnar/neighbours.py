import networkx as nx
import numpy as np

def get_neighbours(adj_matrix: np.ndarray, max_stage: int) -> dict:
    """
    Compute disjoint neighbor sets N^{(j)} for each node up to distance max_stage.

    Parameters:
    -----------
    adj_matrix : np.ndarray
        Adjacency matrix of the network.
    max_stage : int
        Maximum r-stage to compute neighbors for.

    Returns:
    --------
    neighbor_sets : dict[int, dict[int, list[int]]]
        Precompute disjoint neighbour sets N^{(j)} for each node
        i is the node, j is the distance
        neighbour_sets[i][j] = list of nodes at distance j from node i
        eg. neighbour_sets = {0: {1: [1, 2], 2: []}, 1: {1: [0, 2], 2: []}, 2: {1: [0, 1], 2: []}}
        
    """
    G = nx.from_numpy_array(adj_matrix)
    N = adj_matrix.shape[0]
    neighbor_sets = {i: {} for i in range(N)}
    for i in range(N):
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=max_stage)
        for j in range(1, max_stage + 1):
            neighbor_sets[i][j] = [k for k, d in lengths.items() if d == j]
    return neighbor_sets