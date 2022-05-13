import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset
from utils import save, process_control, make_footprint

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    cfg['seed'] = 0
    process_control()
    num_trials = cfg['num_trials']
    num_samples = cfg['num_samples']
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    # data_names = ['MVN', 'GMM', 'RBM']
    data_names = ['MVN']
    params = {k: {} for k in data_names}
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name == 'MVN':
            mean = cfg['mvn']['mean']
            logvar = cfg['mvn']['logvar']
            # ptb_mean = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1, 1.5, 2, 2.5, 3]
            ptb_mean = [0, 0.1]
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_logvar = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'mean': mean, 'logvar': logvar,
                            'ptb_mean': ptb_mean_i, 'ptb_logvar': ptb_logvar}
                dataset = fetch_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
            ptb_logvar = [0, 0.1]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_mean = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'mean': mean, 'logvar': logvar,
                            'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar_i}
                dataset = fetch_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
        elif data_name == 'GMM':
            mean = cfg['gmm']['mean']
            logvar = cfg['gmm']['logvar']
            logweight = cfg['gmm']['logweight']
            ptb_mean = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1, 1.5, 2, 2.5, 3]
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_logvar = float(0)
                ptb_logweight = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples,
                            'mean': mean, 'logvar': logvar, 'logweight': logweight,
                            'ptb_mean': ptb_mean_i, 'ptb_logvar': ptb_logvar, 'ptb_logweight': ptb_logweight}
                dataset = fetch_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
            ptb_logvar = [0, 0.1]
            for i in range(len(ptb_logvar)):
                ptb_mean = float(0)
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_logweight = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples,
                            'mean': mean, 'logvar': logvar, 'logweight': logweight,
                            'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar_i, 'ptb_logweight': ptb_logweight}
                dataset = fetch_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
            ptb_logweight = [0, 0.1]
            for i in range(len(ptb_logweight)):
                ptb_mean = float(0)
                ptb_logvar = float(0)
                ptb_logweight_i = float(ptb_logweight[i])
                params_i = {'num_trials': num_trials, 'num_samples': num_samples,
                            'mean': mean, 'logvar': logvar, 'logweight': logweight,
                            'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar, 'ptb_logweight': ptb_logweight_i}
                dataset = fetch_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
        elif data_name == 'RBM':
            W = cfg['rbm']['W']
            v = cfg['rbm']['v']
            h = cfg['rbm']['h']
            num_iters = cfg['rbm']['num_iters']
            ptb_W = [0, 0.005, 0.007, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035]
            for i in range(len(ptb_W)):
                ptb_W_i = float(ptb_W[i])
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'W': W, 'v': v, 'h': h,
                            'num_iters': num_iters, 'ptb_W': ptb_W_i}
                dataset = fetch_dataset(data_names[i], params_i)
                footprint = make_footprint(params_i)
                params[data_name][footprint] = params_i
        else:
            raise ValueError('Not valid data name')
    save(params, os.path.join('output', 'params', 'params.pkl'))
