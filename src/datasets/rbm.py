import numpy as np
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load


class MVN(Dataset):
    data_name = 'MVN'

    def __init__(self, root, num_trials, num_samples, logweight, mean, logvar, ptb_logweight, ptb_mean, ptb_logvar):
        self.root = os.path.expanduser(root)
        self.num_trials = num_trials
        self.num_samples = num_samples
        self.logweight = logweight
        self.mean = mean
        self.logvar = logvar
        self.ptb_logweight = ptb_logweight
        self.ptb_mean = ptb_mean
        self.ptb_logvar = ptb_logvar
        pivot = [self.logweight, self.mean, self.logvar, self.ptb_logweight, self.ptb_mean, self.ptb_logvar]
        self.pivot_name = hash('_'.join([str(x) for x in pivot]))
        if not check_exists(self.processed_folder):
            self.process()
        self.null, self.alter, self.meta = load(os.path.join(self.processed_folder, 'GMM_{}'.format(self.pivot_name)),
                                                mode='pickle')

    def __getitem__(self, index):
        null, alter = torch.tensor(self.null[index]), torch.tensor(self.alter[index])
        null_param = {'logweight': torch.tensor(self.logweight), 'mean': torch.tensor(self.mean),
                      'logvar': torch.tensor(self.logvar)}
        alter_param = {'logweight': torch.tensor(self.meta['logweight'][index]),
                       'mean': torch.tensor(self.meta['mean'][index]),
                       'logvar': torch.tensor(self.meta['logvar'][index])}
        input = {'null': null, 'alter': alter, 'null_param': null_param, 'alter_param': alter_param}
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        dataset = self.make_data()
        save(dataset, os.path.join(self.processed_folder, 'GMM_{}'.format(self.pivot_name)), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_data(self):
        total_samples = self.num_trials * self.num_samples
        null_normal = torch.distributions.normal.Normal(self.mean, self.logvar.exp().sqrt())
        null = null_normal.sample((total_samples,))
        null_mixture_idx = torch.multinomial(self.logweight.softmax(dim=-1),
                                             num_samples=self.num_trials * self.num_samples,
                                             replacement=True)
        null = torch.gather(null, -1, index=null_mixture_idx.unsqueeze(-1))
        null = null.view(self.num_trials, self.num_samples, -1)
        alter_logweight = self.logweight + self.ptb_logweight * torch.randn((self.num_trials, *self.logweight.size()))
        alter_mean = self.mean + self.mean * torch.randn((self.num_trials, *self.mean.size()))
        alter_logvar = self.logvar + self.ptb_logvar * torch.randn((self.num_trials, *self.logvar.size()))
        alter_normal = torch.distributions.normal.Normal(alter_mean, alter_logvar.exp().sqrt())
        alter = alter_normal.sample((self.num_samples,))
        alter = alter.permute(1, 0, 2)
        alter_mixture_idx = torch.multinomial(alter_logweight.softmax(dim=-1),
                                              num_samples=self.num_samples,
                                              replacement=True)
        alter = torch.gather(alter, -1, index=alter_mixture_idx.unsqueeze(-1))
        meta = {'logweight': alter_logweight, 'mean': alter_mean, 'logvar': alter_logvar}
        return null, alter, meta
