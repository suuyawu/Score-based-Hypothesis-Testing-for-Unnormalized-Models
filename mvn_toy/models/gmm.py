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
            self.model = Normal(mean, logvar.exp().sqrt())
        else:
            self.model = MultivariateNormal(mean, logvar.exp())
        return

    def pdf(self, x, item=False):
        if self.d == 1:
            probs_ = (self.model.log_prob(x.view(-1)).exp() * self.logweight.exp().view(-1, 1))
        else:
            probs_ = (self.model.log_prob(x.unsqueeze(1)).exp() * self.logweight.exp()).transpose(0, 1)
        pdf_ = probs_.sum(dim=0)
        if item:
            # for score and hyvarinen score calculation
            return probs_, pdf_
        else:
            return pdf_

    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        cdf_ = (self.model.cdf(x) * self.logweight.exp().view(-1, 1)).sum(dim=0)
        return cdf_

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()

    def score(self, x):
        if self.d == 1:
            probs_, mpdf = self.pdf(x, item=True)
            mdpdf = (probs_ * (-(x.view(-1) - self.mean) / self.logvar.exp())).sum(dim=0)
            score_ = (mdpdf / mpdf).view(-1, 1)
        else:
            probs_, mpdf = self.pdf(x, item=True)
            mdpdf = (probs_.unsqueeze(-1) * (-1 * torch.matmul((x - self.mean.unsqueeze(1)),
                                                               torch.linalg.inv(self.logvar.exp())))).sum(dim=0)
            score_ = mdpdf / mpdf.unsqueeze(-1)
        return score_

    def hscore(self, x):
        if self.d == 1:
            probs_, mpdf = self.pdf(x, item=True)
            mdpdf = (probs_ * (-(x.view(-1) - self.mean) / self.logvar.exp())).sum(dim=0)
            invcov = self.logvar.exp() ** (-1)
            t1 = ((x - self.mean.unsqueeze(1)) * invcov.unsqueeze(-1) * invcov.unsqueeze(-1)).matmul(
                (x - self.mean.unsqueeze(1)).transpose(-1, -2))
            t2 = - invcov
            t1 = t1.diagonal(dim1=-2, dim2=-1)
            ddpdf = (probs_ * (t1 + t2)).sum(dim=0)
            dlnpdf = mdpdf / mpdf
            hscore_ = -0.5 * (dlnpdf ** 2) + ddpdf / mpdf
        else:
            probs_, mpdf = self.pdf(x, item=True)
            invcov = torch.linalg.inv(self.logvar.exp())
            mdpdf = (probs_.unsqueeze(-1) * (-1 * torch.matmul((x - self.mean.unsqueeze(1)), invcov))).sum(dim=0)
            t1 = (x - self.mean.unsqueeze(1)).matmul(invcov).matmul(invcov).matmul(
                (x - self.mean.unsqueeze(1)).transpose(-1, -2))
            t2 = - invcov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
            t1 = t1.diagonal(dim1=-2, dim2=-1)
            ddpdf = (probs_ * (t1 + t2)).sum(dim=0)
            dlnpdf = mdpdf / mpdf.view(-1, 1)
            hscore_ = (-0.5 * (dlnpdf ** 2)).sum(dim=-1) + ddpdf / mpdf
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
