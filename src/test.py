from config import cfg
from data import fetch_dataset, make_data_loader
import torch
import models
import numpy as np
from pyro.infer import MCMC, NUTS

# if __name__ == "__main__":
#     logweight = torch.log(torch.tensor([0.2, 0.8]))
#     mean = torch.tensor([0., 5.])
#     logvar = torch.tensor([0., 0.])
#
#     ptb_logweight = 0.
#     ptb_mean = 0.1
#     ptb_logvar = 0
#
#     normal = torch.distributions.normal.Normal(mean, logvar.exp())
#     s = normal.sample((1000, 1000))
#     print(s.size())
#     print(s.mean(dim=0))
#     expand_logweight = logweight.view(1, -1).repeat(10, 1)
#     print(expand_logweight.size())
#
#     num_trials = 10
#     num_samples = 20
#     null_normal = torch.distributions.normal.Normal(mean, logvar.exp())
#     null = null_normal.sample((num_trials * num_samples,))
#     print(null.size())
#     mixture_idx = torch.multinomial(logweight.exp(),
#                                     num_samples=num_trials * num_samples,
#                                     replacement=True)
#     print(mixture_idx.size())
#     null = torch.gather(null, -1, index=mixture_idx.unsqueeze(-1))
#     print(null.size())
#     null = null.view(num_trials, num_samples, -1)
#     print(null.size())
#     print(logweight.softmax(dim=0))
#
#     alter_logweight = logweight + ptb_logweight * torch.randn((num_trials, *logweight.size()))
#     alter_mean = mean + mean * torch.randn((num_trials, *mean.size()))
#     alter_logvar = logvar + ptb_logvar * torch.randn((num_trials, *logvar.size()))
#     alter_normal = torch.distributions.normal.Normal(alter_mean, alter_logvar.exp().sqrt())
#     alter = alter_normal.sample((num_samples,))
#     alter = alter.permute(1, 0, 2)
#     print(alter.size())
#     alter_mixture_idx = torch.multinomial(alter_logweight.softmax(dim=-1),
#                                           num_samples=num_samples,
#                                           replacement=True)
#     print(alter_mixture_idx.size())
#     alter = torch.gather(alter, -1, index=alter_mixture_idx.unsqueeze(-1))
#     print(alter.size())

# if __name__ == "__main__":
#     mean = torch.tensor([0., 5.])
#     logvar = torch.tensor([[1., 0.], [0., 1.]]).log()
#     mvn = models.MVN(mean, logvar)
#     true_model = torch.distributions.multivariate_normal.MultivariateNormal(mean, logvar.exp().sqrt())
#     x = true_model.rsample((100,))
#     # x = x.unsqueeze(0).repeat(5, 1, 1)
#     print(x.shape)
#     hscore = mvn.hscore(x)
#     print(hscore.shape)

# if __name__ == "__main__":
#     a = torch.arange(100)
#     n = 15
#     num_chunks = len(a) // n
#     b = torch.split(a, n)
#     if len(a) % n != 0:
#         b = b[:-1]
#     print(len(b), num_chunks)
#     print(b)

# if __name__ == "__main__":
#     ptb = np.linspace(0, 0.1, 20).tolist()
#     print(ptb)

# def unnormalized_pdf_exp(x, power, tau):
#     return torch.exp(-tau * (x['u'] ** power).sum())
#
#



# def make_data_exp(power, tau, num_samples, num_dims):
#     nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, power, tau)))
#     mcmc = MCMC(nuts, num_samples=num_samples, initial_params={'u': torch.zeros((num_dims,))})
#     mcmc.run()
#     samples = mcmc.get_samples()['u']
#     return samples
#
#
# def make_dataset(power, tau, num_samples, num_dims):
#     nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, power, tau)))
#     mcmc = MCMC(nuts, num_samples=num_samples, initial_params={'u': torch.zeros((num_dims,))})
#     mcmc.run()
#     samples = mcmc.get_samples()['u']
#     return samples

# def unnormalized_pdf_normal(x, mean, std):
#     return torch.exp(-0.5 * ((x['u'] - mean) / std) ** 2.)
#
# if __name__ == "__main__":
#     mean = 10
#     std = 2
#     nuts = NUTS(potential_fn=lambda x: -torch.log(unnormalized_pdf_normal(x, mean, std)))
#     mcmc = MCMC(nuts, num_samples=1000, initial_params={'u': torch.zeros((1,))})
#     mcmc.run()
#     samples = mcmc.get_samples()['u']
#     print(samples.mean(), samples.std())

# def unnormalized_pdf_exp_nquad(*x):
#     x_ = np.array(x[:-2])
#     power, tau = x[-2], x[-1]
#     return np.exp(-tau * (x_ ** power).sum())
