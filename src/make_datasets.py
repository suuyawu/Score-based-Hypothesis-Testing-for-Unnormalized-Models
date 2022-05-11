import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset
from utils import save, process_control, process_dataset, makedir_exist_ok

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

num_trials = 1000
num_samples = 500

if __name__ == "__main__":
    data_names = ['GMM']
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name == 'MVN':
            mean = torch.tensor([0., 5.])
            logvar = torch.tensor([[1., 0.1], [0.1, 1.]])
            ptb_mean = 0.1
            ptb_logvar = 0.1
            param = {'num_trials': num_trials, 'num_samples': num_samples,
                     'mean': mean, 'logvar': logvar,
                     'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar}
        elif data_name == 'GMM':
            logweight = torch.log(torch.tensor([0.2, 0.7, 0.1]))
            mean = torch.tensor([[0., 0.], [5., 0.], [1., 2.]])
            logvar = torch.tensor([[[1., 0.1], [0.1, 1.]], [[0.5, 0.1], [0.1, 0.5]], [[0.8, 0.1], [0.1, 0.8]]])
            ptb_logweight = 0.
            ptb_mean = 0.1
            ptb_logvar = 0.1
            param = {'num_trials': num_trials, 'num_samples': num_samples,
                     'mean': mean, 'logvar': logvar, 'logweight': logweight,
                     'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar, 'ptb_logweight': ptb_logweight}
        else:
            raise ValueError('Not valid data name')
        dataset = fetch_dataset(data_names[i], param)
