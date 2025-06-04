import numpy as np
import networkx as nx
import scipy.sparse as sp

def Ymatrix(data, p):
    """
    Returns:
    -----------
    Y : np.ndarray
        Data restricted to p onwards, ie. Y=[y_p+1, ..., y_T]
    """
    return data[:,p:]

def yvector(data):
    """
    Returns:
    -----------
    y : np.ndarray
        Flatten version of Y matrix
    """
    return Ymatrix(data).flatten()

def Zmatrix(data: np.ndarray, p: int) -> np.ndarray:
    """
    Inputs:
    ---------
    data : np.ndarray
        Data matrix of shape (K x T)
        --> we use the convention of Lütkepohl (2005)
    p : int
        Number of lags

    Returns:
    --------
    Z : np.ndarray
        Design matrix of shape  K*p x T-p
    """

    K, T = data.shape
    T_prime = T - p
    Z = np.zeros((K * p, T_prime))

    for lag in range(1, p + 1):
        rows = slice((lag - 1) * K, lag * K)
        Z[rows, :] = data[:, p - lag : T - lag]  # align lagged values

    return Z  # shape: (T - p, K * p)

def Rmatrix(adj_matrix, p, s_tuple, global_alpha=True, global_beta=True):
    """
    Construct restriction matrix R for GNAR restricted VAR estimation.
    
    Parameters:
    -----------
    adj_matrix : np.ndarray
        Adjacency matrix (K x K)
    p : int
        Number of lags
    s_tuple : list of int
        List of s_k values: number of neighbor stages per lag
    global_alpha : bool
        If True, α is shared across all nodes
    global_beta : bool
        If True, β is shared across all nodes

    Returns:
    --------
    R : scipy.sparse.csr_matrix
        Restriction matrix (K^2 * p, d)
    index_map : dict
        Mapping from (i, j, k) to row in vec(B)
    gamma_index_map : dict
        Mapping from (i, k) or (i, k, r) to column index in γ
        E.g., gamma_index_map[(i, k)] = column index for α_i,k
        gamma_index_map[(i, k, r)] = column index for β_i,k,r
    """
    K = adj_matrix.shape[0]
    total_rows = K * K * p # total number of rows in R = number of parameters (ie. A_1, ..., A_p)
    G = nx.from_numpy_array(adj_matrix)
    
    row_idx = []
    col_idx = []
    data = []
    index_map = {}
    gamma_index_map = {} 
    gamma_col_counter = 0 # counts the number of unconstrained parameters (ie. length of gamma)

    for k in range(1, p + 1):
        for i in range(K):  # target node
            for j in range(K):  # source node
                #  α parameters
                if i == j:
                    if global_alpha:
                        key = ('alpha', k)
                        if key not in gamma_index_map:
                            gamma_index_map[key] = gamma_col_counter
                            gamma_col_counter += 1
                        col = gamma_index_map[key]
                    else:
                        key = ('alpha', i, k)
                        if key not in gamma_index_map:
                            gamma_index_map[key] = gamma_col_counter
                            gamma_col_counter += 1
                        col = gamma_index_map[key]

                    row = i + K * ((k - 1) * K + j)
                    row_idx.append(row)
                    col_idx.append(col)
                    data.append(1.0)
                    index_map[(i, j, k)] = row
                    continue

                # For β terms
                try:
                    dist = nx.shortest_path_length(G, source=i, target=j)
                except nx.NetworkXNoPath:
                    continue

                if dist == 0 or dist > s_tuple[k - 1]:
                    continue  # not used at this lag

                if global_beta:
                    key = ('beta', k, dist)
                    if key not in gamma_index_map:
                        gamma_index_map[key] = gamma_col_counter
                        gamma_col_counter += 1
                    col = gamma_index_map[key]
                else:
                    key = ('beta', i, k, dist)
                    if key not in gamma_index_map:
                        gamma_index_map[key] = gamma_col_counter
                        gamma_col_counter += 1
                    col = gamma_index_map[key]

                row = i + K * ((k - 1) * K + j)
                row_idx.append(row)
                col_idx.append(col)
                data.append(1.0)
                index_map[(i, j, k)] = row
    
    # The data just encodes the number of parameters in B
    # Then, these are put into the larger R matrix in positions (row_idx, col_idx)
    R = sp.csr_matrix((data, (row_idx, col_idx)), shape=(total_rows, gamma_col_counter))
    return R, index_map, gamma_index_map



data = np.array([[1,5,2,5,6],[6,8,3,1,3],[7,7,3,6,4],[5,3,1,5,7]])
print(data)
print(Zmatrix(data,1))
A = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]])
p = 2
s = [1, 2]
R, vecB_map, gamma_map = Rmatrix(A, p, s, global_alpha=False, global_beta=True)
print(R.shape)
print(R)
print(R.todense())
print(vecB_map)
print(gamma_map)

