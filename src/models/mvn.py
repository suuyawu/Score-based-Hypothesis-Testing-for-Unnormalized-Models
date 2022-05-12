import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.multivariate_normal import MultivariateNormal


class MVN(nn.Module):
    def __init__(self, mean, logvar):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.model = MultivariateNormal(mean, logvar.exp().sqrt())
    
    def pdf(self, x):
        return self.model.log_prob(x).exp() 
    
    def cdf(self, x):
        return self.model.cdf(x)
    
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

def mvn(mean, logvar):
    model = MVN(mean, logvar)
    return model
