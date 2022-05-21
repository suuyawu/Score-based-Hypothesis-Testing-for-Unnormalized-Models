import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
import os

dpi = 300
cm = 1 / 2.54
width = 80;
height = 20
size = 15
fontsize = 16
# figsize = (width*cm,height*cm)
figsize = (10, 4)


# params = {'legend.fontsize': 'large',
#           'axes.labelsize': size,
#           'axes.titlesize': size,
#           'xtick.labelsize': size*0.75,
#           'ytick.labelsize': size*0.75,
#           'axes.titlepad': 25,
#           'lines.linewidth': 3}
# plt.rcParams.update(params)

def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


base_result_hst = []
base_result_lrt = []
for i in range(10):
    result_path_hst = './output/output{}'.format(i + 1) + '/result/0_MVN_hst-b-g_1.0-0.0_100_0.pt'
    result_path_lrt = './output/output{}'.format(i + 1) + '/result/0_MVN_lrt-b-g_1.0-0.0_100_0.pt'
    base_result_hst.append(load(result_path_hst)['logger'].history['test/Power-t2'])
    base_result_lrt.append(load(result_path_lrt)['logger'].history['test/Power-t2'])

means_hst = []
means_lrt = []
stds_hst = []
stds_lrt = []

for i in range(10):
    means_hst.append(sum(base_result_hst[i]) / len(base_result_hst[i]))
    means_lrt.append(sum(base_result_lrt[i]) / len(base_result_hst[i]))
    stds_hst.append(np.nanstd(np.array(base_result_hst[i])).item())
    stds_lrt.append(np.nanstd(np.array(base_result_lrt[i])).item())

fig, (ax0, ax1) = plt.subplots(figsize=figsize, ncols=2, sharex=True)
ax0.errorbar(np.array([1., 2., 3., 5., 10., 15., 20., 25., 30., 50.]), means_hst, yerr=stds_hst, marker='o',
             label='HST(Simple)', color='red')
ax0.errorbar(np.array([1., 2., 3., 5., 10., 15., 20., 25., 30., 50.]), means_lrt, yerr=stds_lrt, marker='D',
             label='LRT(Simple)', color='black')
# ax0.set_title('Perturbation on the mean')
ax0.set_xlim(50., 0.)
ax0.set_ylim(0.8, 1.)
ax0.set_ylabel('Power on testing mean', fontsize=fontsize)
ax0.set_xlabel("The Maximum Eigen Value $\sigma_{\max}$", fontsize=fontsize)
ax0.xaxis.set_tick_params(labelsize=fontsize)
ax0.yaxis.set_tick_params(labelsize=fontsize)
ax0.grid(linestyle='--', linewidth='0.5')

base_result_hst = []
base_result_lrt = []
for i in range(10):
    result_path_hst = './output/output{}'.format(i + 1) + '/result/0_MVN_hst-b-g_0.0-1.0_100_0.pt'
    result_path_lrt = './output/output{}'.format(i + 1) + '/result/0_MVN_lrt-b-g_0.0-1.0_100_0.pt'
    base_result_hst.append(load(result_path_hst)['logger'].history['test/Power-t2'])
    base_result_lrt.append(load(result_path_lrt)['logger'].history['test/Power-t2'])

means_hst = []
means_lrt = []
stds_hst = []
stds_lrt = []

for i in range(10):
    means_hst.append(sum(base_result_hst[i]) / len(base_result_hst[i]))
    means_lrt.append(sum(base_result_lrt[i]) / len(base_result_hst[i]))
    stds_hst.append(np.nanstd(np.array(base_result_hst[i])).item())
    stds_lrt.append(np.nanstd(np.array(base_result_lrt[i])).item())

print(len(means_hst))
ax1.errorbar(np.array([1., 2., 3., 5., 10., 15., 20., 25., 30., 50.]), means_hst, yerr=stds_hst, marker='o',
             label='HST (Simple)', color='red')
ax1.errorbar(np.array([1., 2., 3., 5., 10., 15., 20., 25., 30., 50.]), means_lrt, yerr=stds_lrt, marker='D',
             label='LRT (Simple)', color='black')
# ax1.set_title('Perturbation on the log covariance')
ax1.set_xlim(50., 0.)
ax1.set_ylim(0.8, 1.)
ax1.set_ylabel('Power on testing covariance', fontsize=fontsize)
ax1.set_xlabel("The Maximum Eigen Value $\sigma_{\max}$", fontsize=fontsize)
ax1.xaxis.set_tick_params(labelsize=fontsize)
ax1.yaxis.set_tick_params(labelsize=fontsize)
ax1.grid(linestyle='--', linewidth='0.5')

# plt.legend(prop={'size': 14}, loc='lower right')
handles, labels = ax1.get_legend_handles_labels()
order = [1, 0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='lower right', fontsize=14)



plt.tight_layout()

save_format = 'pdf'
fig_path = os.path.join('{}.{}'.format('mvn_toy', save_format))
plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()
