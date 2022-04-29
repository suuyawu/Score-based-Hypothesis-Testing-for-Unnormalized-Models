import numpy as np
def find_median_distance(Z):
    """
    return the median of the data distance
    """
    G = np.sum(Z * Z, axis=-1)
    Q = np.expand_dims(G, axis=0)
    R = np.expand_dims(G, axis=1)
    dist = Q + R - 2*np.dot(Z, Z.T)
    dist = np.triu(dist).reshape(-1)
    return np.median(dist[np.where(dist>0)])

def ratio_median_heuristic(Z, score_function):
    """
    return the bandwidth of kernel by ratio of median heuristic
    """
    G = np.sum(Z * Z, axis=-1)
    Q = np.expand_dims(G, axis=0)
    R = np.expand_dims(G, axis=1)
    dist = Q + R - 2*np.dot(Z, Z.T)
    dist = np.triu(dist).reshape(-1)
    dist_median = np.median(dist[np.where(dist>0)])
    
    _zscore = score_function(Z)
    _zscore_sq = np.dot(_zscore, _zscore.T)
    _zscore_sq = np.triu(_zscore_sq, k=1).reshape(-1)
    _zscore_median = np.median(_zscore_sq[np.where(_zscore_sq>0)])
    bandwidth = (dist_median/_zscore_median)**0.25
    return bandwidth

def score_function(x):
    return -x

if __name__ == '__main__':
    #test
    Z = np.arange(1, 91)
    Z.shape = (3,30)
    print(find_median_distance(Z.T))
    print(ratio_median_heuristic(Z.T, score_function))
