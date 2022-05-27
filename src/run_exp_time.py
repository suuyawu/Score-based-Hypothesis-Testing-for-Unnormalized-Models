from config import cfg
from data import fetch_dataset, make_data_loader
import torch
import models
import numpy as np

from pyro.infer import MCMC, NUTS
from scipy import integrate


def unnormalized_pdf_exp(x, power, tau):
    return torch.exp(-tau * (x['u'] ** power).sum())


def unnormalized_pdf_normal(x, mean, std):
    return torch.exp(-0.5 * ((x['u'] - mean) / std) ** 2.)


def make_data_exp(power, tau, num_samples, num_dims):
    nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, power, tau)))
    mcmc = MCMC(nuts, num_samples=num_samples, initial_params={'u': torch.zeros((num_dims,))})
    mcmc.run()
    samples = mcmc.get_samples()['u']
    return samples


def make_dataset(power, tau, num_samples, num_dims):
    nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, power, tau)))
    mcmc = MCMC(nuts, num_samples=num_samples, initial_params={'u': torch.zeros((num_dims,))})
    mcmc.run()
    samples = mcmc.get_samples()['u']
    return samples


# if __name__ == "__main__":
# nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x)))
# mcmc = MCMC(nuts, num_samples=100, initial_params={'u': torch.zeros((1,))})
# mcmc.run()
# print(mcmc.get_samples()['u'].size())

# mean = 1
# var = 1
# nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, mean, var)))
# mcmc = MCMC(nuts, num_samples=100, initial_params={'u': torch.zeros((1,))})
# mcmc.run()
# samples = mcmc.get_samples()['u']
# print(samples.mean(), samples.std())

def unnormalized_pdf_exp_nquad(*x):
    x_ = np.array(x[:-2])
    power, tau = x[-2], x[-1]
    return np.exp(-tau * (x_ ** power).sum())


if __name__ == "__main__":
    power = 4
    tau = 1
    num_dims = 3
    int = integrate.nquad(unnormalized_pdf_exp_nquad, [[-np.infty, np.infty]] * num_dims, args=(power, tau))
    print(int)
