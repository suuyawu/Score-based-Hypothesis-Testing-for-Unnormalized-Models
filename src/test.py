from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

if __name__ == "__main__":
    logweight = torch.log(torch.tensor([0.2, 0.8]))
    mean = torch.tensor([0., 5.])
    logvar = torch.tensor([0., 0.])

    ptb_logweight = 0.
    ptb_mean = 0.1
    ptb_logvar = 0

    normal = torch.distributions.normal.Normal(mean, logvar.exp())
    s = normal.sample((1000, 1000))
    print(s.size())
    print(s.mean(dim=0))
    expand_logweight = logweight.view(1, -1).repeat(10, 1)
    print(expand_logweight.size())

    num_trials = 10
    num_samples = 20
    null_normal = torch.distributions.normal.Normal(mean, logvar.exp())
    null = null_normal.sample((num_trials * num_samples,))
    print(null.size())
    mixture_idx = torch.multinomial(logweight.exp(),
                                    num_samples=num_trials * num_samples,
                                    replacement=True)
    print(mixture_idx.size())
    null = torch.gather(null, -1, index=mixture_idx.unsqueeze(-1))
    print(null.size())
    null = null.view(num_trials, num_samples, -1)
    print(null.size())
    print(logweight.softmax(dim=0))

    alter_logweight = logweight + ptb_logweight * torch.randn((num_trials, *logweight.size()))
    alter_mean = mean + mean * torch.randn((num_trials, *mean.size()))
    alter_logvar = logvar + ptb_logvar * torch.randn((num_trials, *logvar.size()))
    alter_normal = torch.distributions.normal.Normal(alter_mean, alter_logvar.exp().sqrt())
    alter = alter_normal.sample((num_samples,))
    alter = alter.permute(1, 0, 2)
    print(alter.size())
    alter_mixture_idx = torch.multinomial(alter_logweight.softmax(dim=-1),
                                          num_samples=num_samples,
                                          replacement=True)
    print(alter_mixture_idx.size())
    alter = torch.gather(alter, -1, index=alter_mixture_idx.unsqueeze(-1))
    print(alter.size())
