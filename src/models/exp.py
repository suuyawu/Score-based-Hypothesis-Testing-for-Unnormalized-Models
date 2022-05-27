import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from scipy import integrate


def unnormalized_pdf_exp_nquad(*x):
    x_ = np.array(x[:-2])
    power, tau = x[-2], x[-1]
    return np.exp(-tau * (x_ ** power).sum())


class EXP(nn.Module):
    def __init__(self, power, tau, num_dims):
        super().__init__()
        self.reset(power, tau, num_dims)

    def reset(self, power, tau, num_dims):
        self.power = power
        self.tau = nn.Parameter(tau)
        self.num_dims = num_dims
        self.params = {'power': power, 'tau': tau, 'num_dims': num_dims}
        self.normalization_constant = integrate.nquad(unnormalized_pdf_exp_nquad, [[-np.infty, np.infty]] * num_dims,
                                                      args=(power, tau))
        return

    def pdf(self, x):
        if self.d == 1:
            x = x.squeeze(-1)
        pdf_ = self.model.log_prob(x).exp()
        return pdf_

    def score(self, x):
        if self.d == 1:
            score_ = -1 * torch.matmul((x - self.mean), self.logvar.exp() ** (-1)).view(-1, 1)
        else:
            score_ = -1 * torch.matmul((x - self.mean), torch.linalg.inv(self.logvar.exp()))
        return score_

    def hscore(self, x):
        mean = self.mean
        if self.d == 1:
            invcov = self.logvar.exp() ** (-1)
            t1 = 0.5 * ((x - mean) * invcov * invcov).matmul((x - mean).transpose(-1, -2))
            t2 = - invcov
        else:
            invcov = torch.linalg.inv(self.logvar.exp())
            t1 = 0.5 * (x - mean).matmul(invcov).matmul(invcov).matmul((x - mean).transpose(-1, -2))
            t2 = - invcov.diagonal().sum()
        t1 = t1.diagonal(dim1=-2, dim2=-1)
        hscore_ = t1 + t2
        return hscore_

    def fit(self, x):
        raise NotImplementedError
        return


def exp(params):
    power = params['power']
    tau = params['tau']
    num_dims = params['num_dims']
    model = EXP(power, tau, num_dims)
    return model
