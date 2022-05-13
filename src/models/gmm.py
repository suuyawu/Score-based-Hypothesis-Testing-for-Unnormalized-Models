import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import stats


class GMM(nn.Module):
    def __init__(self, mean, logvar, logweight):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.logweight = nn.Parameter(logweight)
        self.d = self.mean.size(-1)
        if self.d == 1:
            self.model = [Normal(_mean, _logvar.exp().sqrt())
                          for (_mean, _logvar) in zip(mean, logvar)]
        else:
            self.model = [MultivariateNormal(_mean, _logvar.exp().sqrt())
                          for (_mean, _logvar) in zip(mean, logvar)]

    def pdf(self, x, item=False):
        _probs = [w * m.log_prob(x).exp()
                  for (w, m) in zip(self.logweight.exp(), self.model)]
        if item:
            # for score and hyvarinen score calculation
            return _probs, sum(_probs)
        else:
            # sum(_probs) in shape (N,)
            return sum(_probs)

    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        _cums = [w * m.cdf(x)
                 for (w, m) in zip(self.logweight.exp(), self.model)]
        # sum (_cums) in shape (N, )
        return sum(_cums)

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()

    def score(self, x):
        _probs, mpdf = self.pdf(x, item=True)
        for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp()):
            print(x.size(), mean.size(), var.size())
            print(_prob * (-(x - mean) / var).size())
        exit()
        mdpdf = sum([_prob * (-(x - mean) / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
        print(mdpdf.size())
        return mdpdf / mpdf

    def hscore(self, x):
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob * (-(x - mean) / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
        ddpdf = sum([_prob * ((-(x - mean) / var) ** 2 - 1 / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
        dlnpdf = mdpdf / mpdf
        hscore = -0.5 * (dlnpdf ** 2) + ddpdf / mpdf
        return hscore


def gmm(params):
    mean = params['mean']
    logvar = params['logvar']
    logweight = params['logweight']
    model = GMM(mean, logvar, logweight)
    return model
