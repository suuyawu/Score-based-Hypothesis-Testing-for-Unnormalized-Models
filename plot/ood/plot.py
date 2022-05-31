import numpy as np
import matplotlib.pyplot as plt
import math
import os

dpi = 300
save_format = 'pdf'


def hist_plot(arrs, labels, n):
    """
    histgram of distribution
    input:
      arrs = [in_score, ood_score]
      labels = [label1, label2]
    """
    color_dict = {'cifar10': 'red', 'tiny-imagenet': 'blue'}
    label_dict = {'cifar10': 'CIFAR10', 'tiny-imagenet': 'Tiny ImageNet'}
    figsize = (5, 4)
    fontsize = 16
    plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    ax_1 = fig.add_subplot(111)
    for (arr, label) in zip(arrs, labels):
        # n = math.ceil((arr.max() - arr.min()))
        if n == 15:
            arr = arr[np.logical_and(arr > -100, arr < -85)]
        ax_1.hist(arr, label=label_dict[label], alpha=0.8, bins=20)
    ax_1.set_xlabel('Test Statistic', fontsize=fontsize)
    ax_1.set_ylabel("Frequency", fontsize=fontsize)
    ax_1.legend(loc="upper right", fontsize=14)
    ax_1.xaxis.set_tick_params(labelsize=fontsize)
    ax_1.yaxis.set_tick_params(labelsize=fontsize)
    ax_1.grid(linestyle='--', linewidth='0.5')
    fig_path = os.path.join('{}.{}'.format('tiny-imagenet_{}'.format(n), save_format))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


def roc_plot(scores, labels):
    """
    roc curve
    input:
      scores=[[id_score1, ood_score1], [id_score2, ood_score2], ...]
      labels = [label1, label2, ...]
    """
    from sklearn.metrics import roc_curve, auc

    color_dict = {'$n=1$': 'blue', '$n=5$': 'orange', '$n=10$': 'green', '$n=15$': 'red', '$n=20$': 'purple'}
    figsize = (5, 4)
    fontsize = 16
    fig = plt.figure(figsize=figsize)
    ax_1 = fig.add_subplot(111)
    for i, score in enumerate(scores):
        score_id, score_ood = score[0], score[1]
        y_true = np.array([0] * len(score_id) + [1] * len(score_ood))
        score_arr = np.append(score_id, score_ood)
        fpr, tpr, _ = roc_curve(y_true, score_arr, pos_label=1)
        roc_auc = auc(fpr, tpr)
        ax_1.plot(fpr, tpr, label=labels[i] + " , AUC=%0.2f" % roc_auc, color = color_dict[labels[i]])
    plt.plot([0, 1], [0, 1], linestyle="--", color='lightgray')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax_1.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax_1.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax_1.xaxis.set_tick_params(labelsize=fontsize)
    ax_1.yaxis.set_tick_params(labelsize=fontsize)
    ax_1.grid(linestyle='--', linewidth='0.5')
    plt.legend(loc="lower right", fontsize=14)
    fig_path = os.path.join('tiny-imagenet_roc.{}'.format(save_format))
    plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    return


sample_sizes = [1, 5, 10, 15, 20]
# sample_sizes = [1, 10, 15]

rocs = []
labels = []
for n in sample_sizes:
    ood_path = './' + 'tiny-imagenet' + '_hscore_fd_{}.npy'.format(n)
    ood_arr = np.load(ood_path)

    id_path = './' + 'cifar10_' + 'tiny-imagenet' + '_hscore_fd_{}.npy'.format(n)
    id_arr = np.load(id_path)
    if n == 1:
        ood_arr = ood_arr*32*32*np.log(2)
        id_arr = id_arr*32*32*np.log(2)
    hist_plot([id_arr, ood_arr], ['cifar10', 'tiny-imagenet'], n)
    rocs.append([id_arr, ood_arr])
    labels.append('$n={}$'.format(n))

roc_plot(rocs, labels)
