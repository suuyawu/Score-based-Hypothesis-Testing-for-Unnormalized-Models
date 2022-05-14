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
num_samples = 1000
vis_path = os.path.join('output', 'vis')


def vis(null_model, alter_model, null, alter, data_name):
    with torch.no_grad():
        null_x, alter_x = null[:, 0], alter[:, 0]
        null_y, alter_y = null[:, 1], alter[:, 1]
        range = 40
        x_axis, y_axis = np.mgrid[-range:range:.1, -range:range:.1]
        pos = torch.from_numpy(np.dstack((x_axis, y_axis)).astype(np.float32))
        null_z = null_model.pdf(pos).detach().log2()
        alter_z = alter_model.pdf(pos).detach().log2()
        fig = plt.figure(figsize=(10, 6))
        ax_1 = fig.add_subplot(121)
        ax_1.contourf(x_axis, y_axis, null_z)
        ax_1.scatter(null_x, null_y, s=1, color='r')
        ax_2 = fig.add_subplot(122)
        ax_2.contourf(x_axis, y_axis, alter_z)
        ax_2.scatter(alter_x, alter_y, s=1, color='blue')
        fig.tight_layout()
        dir_path = os.path.join(vis_path, 'test')
        fig_path = os.path.join(dir_path, '{}.png'.format(data_name))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close()
    return


if __name__ == "__main__":
    cfg['seed'] = 0
    process_control()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    data_names = ['MVN', 'GMM', 'RBM']
    for i in range(len(data_names)):
        data_name = data_names[i]
        if data_name == 'MVN':
            mean = cfg['mvn']['mean']
            logvar = cfg['mvn']['logvar']
            ptb_mean = float(1)
            ptb_logvar = float(0.1)
            params = {'num_trials': num_trials, 'num_samples': num_samples, 'mean': mean, 'logvar': logvar,
                      'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar}
        elif data_name == 'GMM':
            mean = cfg['gmm']['mean']
            logvar = cfg['gmm']['logvar']
            logweight = cfg['gmm']['logweight']
            ptb_logweight = float(0)
            ptb_mean = float(5)
            ptb_logvar = float(0.1)
            params = {'num_trials': num_trials, 'num_samples': num_samples,
                      'mean': mean, 'logvar': logvar, 'logweight': logweight,
                      'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar, 'ptb_logweight': ptb_logweight}
        elif data_name == 'RBM':
            dim_v = 2
            dim_h = 10
            generator = torch.Generator()
            generator.manual_seed(cfg['seed'])
            W = torch.randn(dim_v, dim_h, generator=generator)
            v = torch.randn(dim_v, generator=generator)
            h = torch.randn(dim_h, generator=generator)
            ptb_W = float(1)
            num_iters = int(1000)
            params = {'num_trials': num_trials, 'num_samples': num_samples,
                      'W': W, 'v': v, 'h': h, 'ptb_W': ptb_W, 'num_iters': num_iters}
        else:
            raise ValueError('Not valid data name')
        dataset = fetch_dataset(data_names[i], params)
        data_loader = make_data_loader(dataset, 'gof')
        input = next(iter(data_loader['test']))
        input = collate(input)
        if data_name == 'MVN':
            null_model = models.mvn(input['null_param'])
            alter_model = models.mvn(input['alter_param'])
        elif data_name == 'GMM':
            null_model = models.gmm(input['null_param'])
            alter_model = models.gmm(input['alter_param'])
        elif data_name == 'RBM':
            null_model = models.rbm(input['null_param'])
            alter_model = models.rbm(input['alter_param'])
        else:
            raise ValueError('Not valid data name')
        vis(null_model, alter_model, input['null'], input['alter'], data_name)
