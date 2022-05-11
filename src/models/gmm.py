import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class GMM(nn.Module):
    def __init__(self, mean, logvar, logweight):
        super().__init__()
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.logweight = nn.Parameter(logweight)

    def pdf(self, x, item=False):
        _probs = [w * torch.exp(m.log_prob(x))
                  for (w, m) in zip(self.weight, self.model)]
        if item:
            # for score and hyvarinen score calculation
            return _probs, sum(_probs)
        else:
            # sum(_probs) in shape Nx1
            return sum(_probs).sum(-1)

    def cdf(self, x):
        x = x.squeeze(-1)
        _cums = [w * m.cdf(x)
                 for (w, m) in zip(self.weight, self.model)]
        return sum(_cums)

    def cdf_numpy(self, x):
        x = x.squeeze(-1)
        mcdf = 0.0
        for i in range(len(self.weight)):
            mcdf += self.weight[i].cpu().numpy() * stats.norm.cdf(x, loc=self.mean[i].cpu().numpy(),
                                                                  scale=np.sqrt(np.exp(self.logvar[i].cpu().numpy())))
        return mcdf

    def score(self, x):
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob * (-(x - mean) / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        return mdpdf / mpdf

    def hscore(self, x):
        x = x.squeeze(-1)
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob * (-(x - mean) / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        ddpdf = sum([_prob * ((-(x - mean) / var) ** 2 - 1 / var)
                     for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        dlnpdf = mdpdf / mpdf
        hscore = -0.5 * (dlnpdf ** 2) + ddpdf / mpdf
        return hscore


def gmm(mean, logvar, logweight):
    model = GMM(mean, logvar, logweight)
    return model
