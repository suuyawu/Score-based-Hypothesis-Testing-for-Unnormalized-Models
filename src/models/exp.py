import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from scipy import integrate


def unnormalized_pdf_exp_nquad(*x):
    x_ = np.array(x[:-2])
    power, tau = x[-2], x[-1]
    d_ = len(x_)
    if d_ == 1:
        u_pdf = np.exp(-tau * (x_[0] ** power))
    elif d_ == 2:
        u_pdf = np.exp(-tau * (x_[0] ** power + x_[1] ** power +
                               (x_[0] * x_[1]) ** (power / 2)))
    elif d_ == 3:
        u_pdf = np.exp(-tau * (x_[0] ** power +
                               x_[1] ** power +
                               x_[2] ** power +
                               (x_[0] * x_[1]) ** (power / 2) +
                               (x_[0] * x_[2]) ** (power / 2) +
                               (x_[1] * x_[2]) ** (power / 2)))
    elif d_ == 4:
        u_pdf = np.exp(-tau * (x_[0] ** power +
                               x_[1] ** power +
                               x_[2] ** power +
                               x_[3] ** power +
                               (x_[0] * x_[1]) ** (power / 2) +
                               (x_[0] * x_[2]) ** (power / 2) +
                               (x_[0] * x_[3]) ** (power / 2) +
                               (x_[1] * x_[2]) ** (power / 2) +
                               (x_[1] * x_[3]) ** (power / 2) +
                               (x_[2] * x_[3]) ** (power / 2)))
    else:
        raise ValueError('Not valid d')
    return u_pdf


class EXP(nn.Module):
    def __init__(self, power, tau, num_dims):
        super().__init__()
        self.normalization_constant = None
        self.reset(power, tau, num_dims)

    def reset(self, power, tau, num_dims):
        self.register_buffer('power', power)
        self.tau = nn.Parameter(tau)
        self.register_buffer('num_dims', num_dims)
        self.params = {'power': power, 'tau': tau, 'num_dims': num_dims}
        if 'lrt' in cfg['test_mode']:
            nc_name = 'nc_{}_{}'.format(self.tau.data.item(), num_dims.item())
            if nc_name not in cfg:
                cfg[nc_name] = integrate.nquad(unnormalized_pdf_exp_nquad,
                                               [[-np.infty, np.infty]] * num_dims.item(),
                                               args=(power.cpu().numpy(), tau.data.cpu().numpy()))[0]
            self.normalization_constant = cfg[nc_name]
        return

    def pdf(self, x):
        pdf_ = self.normalization_constant ** (-1) * torch.exp(-(self.tau * (x ** self.power).sum(-1)))
        return pdf_

    def score(self, x):
        score_ = -self.power * self.tau * (x ** (self.power - 1))
        return score_

    def hscore(self, x):
        # self.power >= 2
        term1 = self.score(x)
        term2 = -(self.power * (self.power - 1)) * self.tau * (x ** (self.power - 2))
        hscore_ = (0.5 * term1 ** 2 + term2).sum(-1)
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
