import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class MVN(nn.Module):
    def __init__(self, mean, logvar):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.d = self.mean.size(-1)
        if self.d == 1:
            self.model = Normal(mean, logvar.exp().sqrt())
        else:
            self.model = MultivariateNormal(mean, logvar.exp().sqrt())
    
    def pdf(self, x):
        return self.model.log_prob(x).exp()
    
    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        return self.model.cdf(x)

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()
    
    def score(self, x):
        return -1 * torch.matmul((x - self.mean), torch.linalg.inv(self.logvar.exp()))

    def hscore(self, x):
        invcov = torch.linalg.inv(self.logvar.exp())
        t1 = 0.5 * torch.matmul(torch.matmul(torch.matmul((x - self.mean), invcov), invcov),
                                (x - self.mean).transpose())
        t2 = - invcov.diagonal().sum()
        t1 = t1.diagonal().sum()
        hs = t1 / x.shape[0] + t2
        return hs

def mvn(params):
    mean = params['mean']
    logvar = params['logvar']
    model = MVN(mean, logvar)
    return model
