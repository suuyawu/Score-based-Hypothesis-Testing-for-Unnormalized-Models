from config import cfg
from data import fetch_dataset, make_data_loader
import os
import time
import torch
import models
import numpy as np
from utils import process_control, makedir_exist_ok, collate, save, load
from pyro.infer import MCMC, NUTS
from scipy import integrate
from modules import GoodnessOfFit
import matplotlib.pyplot as plt

save_format = 'pdf'
vis_path = os.path.join('output', 'vis', save_format)
dpi = 300


def run_exp_time():
    num_trials = 4
    num_samples = 10000
    power = torch.tensor([4.])
    tau = torch.tensor([1.])
    ptb_tau = float(1.0)
    # num_dims_list = [1, 2, 3, 4]
    num_dims_list = [1, 2, 3]
    time_lrt = []
    time_hst = []
    for i in range(len(num_dims_list)):
        print('Dimension: {}'.format(num_dims_list[i]))
        num_dims = torch.tensor([num_dims_list[i]])
        params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'power': power, 'tau': tau,
                    'num_dims': num_dims, 'ptb_tau': ptb_tau}
        dataset = fetch_dataset(cfg['data_name'], params_i)
        data_loader = make_data_loader(dataset, 'gof')
        cfg['test_mode'] = 'lrt-b-g'
        time_lrt_i = []
        for j, input in enumerate(data_loader['test']):
            nc_name = 'nc_{}_{}'.format(tau.item(), num_dims.item())
            if nc_name in cfg:
                del cfg[nc_name]
            s = time.time()
            gof = GoodnessOfFit(cfg['test_mode'], cfg['alter_num_samples'], cfg['alter_noise'])
            input = collate(input)
            gof.test(input)
            time_lrt_i_j = time.time() - s
            time_lrt_i.append(time_lrt_i_j)
        time_lrt.append(time_lrt_i)
        cfg['test_mode'] = 'hst-b-g'
        time_hst_i = []
        for j, input in enumerate(data_loader['test']):
            s = time.time()
            gof = GoodnessOfFit(cfg['test_mode'], cfg['alter_num_samples'], cfg['alter_noise'])
            input = collate(input)
            gof.test(input)
            time_hst_i_j = time.time() - s
            time_hst_i.append(time_hst_i_j)
        time_hst.append(time_hst_i)
    result = {'num_dims_list': num_dims_list, 'time_lrt': time_lrt, 'time_hst': time_hst}
    save(result, os.path.join('output', 'result', 'exp_time.pt'))
    return


def plot(num_dims_list, time_lrt, time_hst):
    color_dict = {'ksd-u': 'blue', 'ksd-v': 'cyan', 'lrt-b-g': 'black', 'lrt-b-e': 'gray', 'hst-b-g': 'red',
                  'hst-b-e': 'orange', 'mmd': 'green'}
    linestyle_dict = {'ksd-u': '-', 'ksd-v': '--', 'lrt-b-g': '-', 'lrt-b-e': '--', 'hst-b-g': '-',
                      'hst-b-e': '--', 'mmd': '-'}
    label_dict = {'ksd-u': 'KSD-U', 'ksd-v': 'KSD-V', 'lrt-b-g': 'LRT (Simple)', 'lrt-b-e': 'LRT (Composite)',
                  'hst-b-g': 'HST (Simple)', 'hst-b-e': 'HST (Composite)', 'mmd': 'MMD'}
    marker_dict = {'ksd-u': 'X', 'ksd-v': 'x', 'lrt-b-g': 'D', 'lrt-b-e': 'd',
                   'hst-b-g': 'o', 'hst-b-e': '^', 'mmd': 's'}
    label_loc_dict = {'time': 'upper left'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    capsize = 3
    capthick = 3
    time_lrt = np.array(time_lrt)
    time_hst = np.array(time_hst)
    time_lrt_mean = time_lrt.mean(axis=-1)
    time_lrt_std = time_lrt.std(axis=-1)
    time_hst_mean = time_hst.mean(axis=-1)
    time_hst_std = time_hst.std(axis=-1)
    fig = plt.figure(figsize=figsize)
    ax_1 = fig.add_subplot(111)
    label = 'lrt-b-g'
    ax_1.errorbar(num_dims_list, time_lrt_mean, yerr=time_lrt_std / 2, color=color_dict[label],
                  linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label])
    label = 'hst-b-g'
    ax_1.errorbar(num_dims_list, time_hst_mean, yerr=time_hst_std / 2, color=color_dict[label],
                  linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label])
    # label = 'lrt-b-g'
    # ax_1.plot(num_dims_list, time_lrt_mean, color=color_dict[label],
    #               linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label])
    # label = 'hst-b-g'
    # ax_1.plot(num_dims_list, time_hst_mean, color=color_dict[label],
    #               linestyle=linestyle_dict[label], label=label_dict[label], marker=marker_dict[label])
    ax_1.set_xlabel('Dimension', fontsize=fontsize['label'])
    ax_1.set_ylabel('CPU Time (s)', fontsize=fontsize['label'])
    ax_1.set_xticks(num_dims_list)
    ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
    ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
    ax_1.legend(loc=label_loc_dict['time'], fontsize=fontsize['legend'])
    ax_1.grid(linestyle='--', linewidth='0.5')
    ax_1.set_yscale('log')
    plt.tight_layout()
    dir_path = os.path.join(vis_path, 'time')
    fig_path = os.path.join(dir_path, '{}.{}'.format(cfg['data_name'], save_format))
    makedir_exist_ok(dir_path)
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


if __name__ == "__main__":
    cfg['seed'] = 0
    cfg['control']['data_name'] = 'EXP'
    cfg['control']['model_name'] = 'exp'
    cfg['device'] = 'cpu'
    process_control()
    # run_exp_time()
    result = load(os.path.join('output', 'result', 'exp_time.pt'))
    num_dims_list = result['num_dims_list']
    time_lrt = result['time_lrt']
    time_hst = result['time_hst']
    plot(num_dims_list, time_lrt, time_hst)
