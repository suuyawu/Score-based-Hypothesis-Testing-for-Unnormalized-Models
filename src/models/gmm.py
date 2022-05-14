import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.mixture import GaussianMixture


class GMM(nn.Module):
    def __init__(self, mean, logvar, logweight):
        super().__init__()
        self.reset(mean, logvar, logweight)

    def reset(self, mean, logvar, logweight):
        self.mean = nn.Parameter(mean)
        self.logvar = nn.Parameter(logvar)
        self.logweight = nn.Parameter(logweight)
        self.params = {'mean': mean, 'logvar': logvar, 'logweight': logweight}
        self.d = self.mean.size(-1)
        if self.d == 1:
            self.model = [Normal(_mean, _logvar.exp().sqrt())
                          for (_mean, _logvar) in zip(mean, logvar)]
        else:
            self.model = [MultivariateNormal(_mean, _logvar.exp())
                          for (_mean, _logvar) in zip(mean, logvar)]
        return

    def pdf(self, x, item=False):
        # if self.d == 1 and x.dim() == 3:
        #     x = x.squeeze(-1)
        _probs = [w * m.log_prob(x).exp()
                  for (w, m) in zip(self.logweight.exp(), self.model)]
        if item:
            # for score and hyvarinen score calculation
            return _probs, sum(_probs)
        else:
            # sum(_probs) in shape (N,)
            pdf_ = sum(_probs)
            return pdf_

    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        _cums = [w * m.cdf(x)
                 for (w, m) in zip(self.logweight.exp(), self.model)]
        cdf_ = sum(_cums)
        return cdf_

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()

    def score(self, x):
        if self.d == 1:
            _probs, mpdf = self.pdf(x, item=True)
            mdpdf = sum([_prob * (-(x - mean) / var)
                         for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
            score_ = mdpdf / mpdf
        else:
            _probs, mpdf = self.pdf(x, item=True)
            mdpdf = 0
            for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp()):
                mdpdf += _prob.view(-1, 1) * (-(x - mean).matmul(torch.linalg.inv(var)))
            score_ = mdpdf / mpdf
        return score_

    def hscore(self, x):
        if self.d == 1:
            _probs, mpdf = self.pdf(x, item=True)
            mdpdf = sum([_prob * (-(x - mean) / var)
                         for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
            ddpdf = sum([_prob * ((-(x - mean) / var) ** 2 - 1 / var)
                         for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
            dlnpdf = mdpdf / mpdf
            hscore_ = -0.5 * (dlnpdf ** 2) + ddpdf / mpdf
        else:
            _probs, mpdf = self.pdf(x, item=True)
            mdpdf = 0
            for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp()):
                mdpdf += _prob.view(-1, 1) * (-(x - mean).matmul(torch.linalg.inv(var)))
            ddpdf = 0
            for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp()):
                ddpdf += _prob.view(-1, 1) * (-(x - mean).matmul(torch.linalg.inv(var)))

            mdpdf = sum([_prob * (-(x - mean) / var)
                         for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
            ddpdf = sum([_prob * ((-(x - mean) / var) ** 2 - 1 / var)
                         for (_prob, mean, var) in zip(_probs, self.mean, self.logvar.exp())])
            dlnpdf = mdpdf / mpdf
            hscore_ = -0.5 * (dlnpdf ** 2) + ddpdf / mpdf
        return hscore_

    def fit(self, x):
        gm = GaussianMixture(n_components=cfg['gmm']['num_components'], random_state=cfg['seed']).fit(x.cpu().numpy())
        mean = x.new_tensor(gm.means_)
        logvar = x.new_tensor(gm.covariances_).log()
        if self.d == 1:
            logvar = logvar.squeeze(-1)
        logweight = x.new_tensor(gm.weights_).log()
        self.reset(mean, logvar, logweight)
        return


def gmm(params):
    mean = params['mean']
    logvar = params['logvar']
    logweight = params['logweight']
    model = GMM(mean, logvar, logweight)
    return model
