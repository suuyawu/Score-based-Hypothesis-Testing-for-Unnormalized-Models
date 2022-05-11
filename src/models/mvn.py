import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class MVN(nn.Module):
    def __init__(self, mean, logvar):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)

    def dlnprob(self, x):
        return -1 * torch.matmul((x - self.mean), torch.linalg.inv(self.cov))

    def MGprob(self, x):
        from scipy.stats import multivariate_normal
        mvn = multivariate_normal(mean=self.mean, cov=self.cov)
        return mvn.pdf(x)

    def lnprob(self, x):
        return torch.mean(torch.log(self.MGprob(x)))

    def hscore(self, x):
        invcov = torch.linalg.inv(self.cov)
        t1 = 0.5 * torch.matmul(torch.matmul(torch.matmul((x - self.mean), invcov), invcov),
                                (x - self.mean).transpose())
        t2 = - invcov.diagonal().sum()
        t1 = t1.diagonal().sum()
        hs = t1 / x.shape[0] + t2
        return hs


def mvn(mean, logvar):
    model = MVN(mean, logvar)
    return model
