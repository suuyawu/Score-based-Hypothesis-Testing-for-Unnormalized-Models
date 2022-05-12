import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import stats

class GMM(nn.Module):
    def __init__(self, mean, logvar, logweight):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.logweight = nn.Parameter(logweight)
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
        _cums = [w * m.cdf(x)
                 for (w, m) in zip(self.logweight.exp(), self.model)]
        # sum (_cums) in shape (N, )
        return sum(_cums)

    def cdf_numpy(self, x):
        # numpy computation for ks test and cramer-von mises tests
        mcdf = 0.0
        mean_double_numpy = self.mean.type(torch.float64).cpu().numpy()
        logvar_double_numpy = self.logvar.type(torch.float64).cpu().numpy()
        logweight_double_numpy = self.logweight.type(torch.float64).cpu().numpy()

        for i in range(len(logweight_double_numpy)):
            mcdf += logweight_double_numpy.exp()[i] * stats.multivariate_normal.cdf(x, loc=mean_double_numpy[i],
                                                                  scale=np.sqrt(np.exp(logvar_double_numpy[i])))
        return mcdf

    def score(self, x):
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob * (-(x - mean) / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
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

def gmm(mean, logvar, logweight):
    model = GMM(mean, logvar, logweight)
    return model
