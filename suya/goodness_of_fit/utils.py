import numpy as np
import torch

def find_median_distance(Z, T):
    """return the median of the data distance
    Args:
        Z (Tensor in size (num_samples, dim)): the samples 
    """
    Q = torch.sum(Z * Z, dim=-1).unsqueeze(0)
    R = torch.sum(T * T, dim=-1).unsqueeze(1)
    dist = Q + R - 2*torch.matmul(Z, T.t())
    dist=torch.triu(dist).view(-1)
    return torch.median(dist[dist>0.])

def ratio_median_heuristic(Z, score_func):
    """return the bandwidth of kernel by ratio of median heuristic
    Args:
        Z (Tensor in size (num_samples, dim)): the samples 
        score_func (func): true grad_x_p(x)
    """
    G = torch.sum(Z * Z, dim=-1)
    Q = torch.unsqueeze(G, dim=0)
    R = torch.unsqueeze(G, dim=1)
    dist = Q + R - 2*torch.matmul(Z, Z.t())
    dist = torch.triu(dist).view(-1)
    dist_median = torch.median(dist[dist>0.])
    
    _zscore = score_func(Z)
    _zscore_sq = torch.matmul(_zscore, _zscore.t())
    _zscore_sq = torch.triu(_zscore_sq, diagonal=1).reshape(-1)
    _zscore_median = torch.median(_zscore_sq[_zscore_sq>0.])
    bandwidth = (dist_median/_zscore_median)**0.25
    return bandwidth

if __name__ == '__main__':
    #test
    Z = torch.tensor(np.arange(1, 91))
    Z = Z.view(3,30)
    def score_function(x):
        return -x
    print(find_median_distance(Z.t(), Z.t()))
    print(ratio_median_heuristic(Z.t(), score_function))
