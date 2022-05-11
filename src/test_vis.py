import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
import matplotlib.pyplot as plt
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from utils import save, process_control, makedir_exist_ok, collate

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

num_trials = 2
num_samples = 500
vis_path = os.path.join('output', 'vis')


def vis(null, alter, data_name):
    null_x, alter_x = null[:, 0], alter[:, 0]
    null_y, alter_y = null[:, 1], alter[:, 1]
    x_axis, y_axis = np.mgrid[-10:10:.1, -10:10:.1]
    pos = torch.from_numpy(np.dstack((x_axis, y_axis)))
    # null_z = null_model.pdf(pos))
    # alter_z = alter_model.pdf(pos)
    fig = plt.figure()
    ax_1 = fig.add_subplot(121)
    # ax.contourf(x_axis, y_axis, z)
    ax_1.scatter(null_x, null_y, s=1, color='r')
    ax_2 = fig.add_subplot(122)
    # ax.contourf(x_axis, y_axis, z)
    ax_2.scatter(alter_x, alter_y, s=1, color='blue')
    fig.tight_layout()
    dir_path = os.path.join(vis_path, 'test')
    fig_path = os.path.join(dir_path, '{}.png'.format(data_name))
    makedir_exist_ok(dir_path)
    plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


if __name__ == "__main__":
    process_control()
    cfg['seed'] = 0
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    data_names = ['RBM']
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name == 'MVN':
            mean = torch.tensor([0., 5.])
            logvar = torch.tensor([[1., 0.1], [0.1, 1.]])
            ptb_mean = 0.1
            ptb_logvar = 0.1
            params = {'num_trials': num_trials, 'num_samples': num_samples,
                      'mean': mean, 'logvar': logvar,
                      'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar}
        elif data_name == 'GMM':
            logweight = torch.log(torch.tensor([0.2, 0.7, 0.1]))
            mean = torch.tensor([[0., 0.], [5., 0.], [1., 2.]])
            logvar = torch.tensor([[[1., 0.1], [0.1, 1.]], [[0.5, 0.1], [0.1, 0.5]], [[0.8, 0.1], [0.1, 0.8]]])
            ptb_logweight = 0.
            ptb_mean = 0.1
            ptb_logvar = 0.1
            params = {'num_trials': num_trials, 'num_samples': num_samples,
                      'mean': mean, 'logvar': logvar, 'logweight': logweight,
                      'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar, 'ptb_logweight': ptb_logweight}
        elif data_name == 'RBM':
            dim_v = 50
            dim_h = 40
            W = torch.randn(dim_v, dim_h)
            v = torch.randn(dim_v)
            h = torch.randn(dim_h)
            ptb_W = 0.1
            num_iters = 200
            params = {'num_trials': num_trials, 'num_samples': num_samples,
                      'W': W, 'v': v, 'h': h, 'ptb_W': ptb_W, 'num_iters': num_iters}
        else:
            raise ValueError('Not valid data name')
        dataset = fetch_dataset(data_names[i], params)
        data_loader = make_data_loader(dataset, 'ht')
        input = next(iter(data_loader['test']))
        input = collate(input)
        vis(input['null'][0].numpy(), input['alter'][0].numpy(), data_name)
