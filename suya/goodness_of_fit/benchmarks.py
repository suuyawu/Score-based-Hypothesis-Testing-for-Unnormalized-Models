"""Benchmarks
Standard methods:
Goodness of fit for comparing two models (Simple vs. Simple test): Likelihood Ratio Test (LRT)
Goodness of fit: Kolmogorov Smirnov (KS) test and Cramér-von Mises (CM) test

KSD-based methods:
Goodness of fit: KSD [1]
Two samples tests: Sliced KSD [2] 
[1] Liu, Q., Lee, J. D., & Jordan, M. (2016). A kernelized Stein discrepancy for goodness-of-fit tests. In 33rd International Conference on Machine Learning, ICML 2016 (Vol. 1).
[2] Gong, W., Li, Y., & Hernández-Lobato, J. M. (2020). Sliced Kernelized Stein Discrepancy. http://arxiv.org/abs/2006.16531

Two sample tests: MMD [3] and NTK-MMD [4]
[3] Gretton, A., Borgwardt, K. M., Rasch, M. J., Smola, A., Schölkopf, B., & Smola GRETTON, A. (2012). A Kernel Two-Sample Test Bernhard Schölkopf. In Journal of Machine Learning Research (Vol. 13). www.gatsby.ucl.ac.uk/
[4] Cheng, X., & Xie, Y. (2021). Neural Tangent Kernel Maximum Mean Discrepancy. http://arxiv.org/abs/2106.03227
"""
import time
import numpy as np
import torch
from scipy import stats
from utils import *
import matplotlib.pyplot as plt


##CM test is not in the scipy 1.5.2, KS test needs to be checked
def Nonparametric_test_batch(test_name, alter_samples, model0, alpha=0.05):
    """Perform Cramér-von Mises test or Kolmogorov-Smirnov test"""
    num_tests = alter_samples.shape[0]
    alter_statisitics = []
    pvalues = []
    start_time = time.time()
    alter_samples = alter_samples.cpu().numpy()
    for i in range(num_tests):
        if test_name == 'Kolmogorov Smirnov':
            ks = stats.kstest(alter_samples[i], model0.cdf_numpy, alternative='two-sided', mode='auto')
            test_statistic = ks[0]
            pvalue = ks[1]
        if test_name == 'Cramér-von Mises':
            cm = stats.cramervonmises(alter_samples[i], model0.cdf_numpy)
            pvalue = cm.pvalue
            test_statistic = cm.statistic
        pvalues.append(pvalue)
        alter_statisitics.append(test_statistic)
    mean_time = (time.time() - start_time) / num_tests
    result_batch = {
        'critical value': None,
        'statisitics': alter_statisitics,
        'pvalues': pvalues,
        'power': sum([pvalue < alpha for pvalue in pvalues]) / num_tests,
        'time': mean_time
    }
    return result_batch


def KSD_statistics(y, score_func, width='heuristic', V_stat=False):
    """KSD for goodness of fit test pytorch implementation of https://rdrr.io/cran/KSD/"""
    # set up the the bandwidth of RBF Kernel
    if width == 'heuristic':
        h = ratio_median_heuristic(y, score_func)
    else:
        h = torch.sqrt(0.5 * find_median_distance(y, y))

    # RBF kernel
    n_samples, d_samples = y.size()
    if d_samples is None:
        y = y.unsqueeze(1)
    _yscore = score_func(y)
    yy = torch.matmul(y, y.t())
    ys = torch.unsqueeze(torch.sum(y * y, dim=-1), dim=1)
    y_yscore = torch.sum(y * _yscore, dim=1)
    H = ys + ys.t() - 2 * yy
    Kxy = torch.exp(-H / (2 * h ** 2))

    # Calculate gradient
    sqxdy = -(torch.matmul(_yscore, y.t()) - torch.unsqueeze(y_yscore, dim=1)) / h ** 2
    dxsqy = sqxdy.t()
    dxdy = (-H / h ** 4 + d_samples / h ** 2)

    M = (torch.matmul(_yscore, _yscore.t()) + sqxdy + dxsqy + dxdy) * Kxy
    M2 = M - torch.diag(torch.diag(M))
    if V_stat:
        return M, torch.sum(M) / (n_samples ** 2)
    else:
        return M2, torch.sum(M2) / (n_samples * (n_samples - 1))


def multinomial_weights(num_bootstrap, num_samples):
    """Sample multinomial weights for bootstrap by Huskova & Janssen (1993)"""
    weights = np.random.multinomial(num_samples, np.ones(num_samples) / num_samples, size=int(num_bootstrap))
    weights = weights / num_samples
    weights = torch.from_numpy(weights)

    weights_exp1 = torch.unsqueeze(weights, dim=-1)  # m x N x 1
    weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x N
    return weights_exp1, weights_exp2


def multinomial_bootstrap(test_name, num_bootstrap, num_samples, null_samples, null_score):
    """Bootstrap algorithm for U-statistics by Huskova & Janssen (1993)"""
    if test_name == 'KSD-U':
        null_items, _ = KSD_statistics(null_samples[:num_samples], null_score, V_stat=False)
    if test_name == 'KSD-V':
        null_items, _ = KSD_statistics(null_samples[:num_samples], null_score, V_stat=True)
    weights_exp1, weights_exp2 = multinomial_weights(num_bootstrap, num_samples)
    null_items = torch.unsqueeze(null_items, dim=0)  # 1 x N x N
    bootstrap_null_samples = (weights_exp1 - 1. / num_samples) * null_items * (
                weights_exp2 - 1. / num_samples)  # m x N x N
    if test_name == 'KSD-U':
        bootstrap_null_samples = torch.sum(torch.sum(bootstrap_null_samples, dim=-1), dim=-1)
    if test_name == 'KSD-V':
        bootstrap_null_samples = torch.sum(torch.sum(bootstrap_null_samples, dim=-1), dim=-1)
    return bootstrap_null_samples


def KSD_U_test(alter_samples, bootstrap_null_samples, null_score):
    _, test_statistic = KSD_statistics(alter_samples, null_score, V_stat=False)
    pvalue = torch.mean((bootstrap_null_samples >= test_statistic).float())
    return test_statistic, pvalue


def l2norm_dist(sample_1, sample_2):
    """Compute the matrix of all squared pairwise distances.
    
    Args:
        sample_1 (Tensor in shape (n_1, d)): The first sample
        sample_2 (Tensor in shape (n_2, d)): The second sample

    Returns
        Tensor in shape (n_1, n_2): The [i, j]-th entry is equal to ``|| sample_1[i, :] - sample_2[j, :] ||_p``.
    """
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
    norms = (norms_1.expand(n_1, n_2) +
             norms_2.transpose(0, 1).expand(n_1, n_2))
    distances_squared = norms - 2 * sample_1.mm(sample_2.t())
    return torch.sqrt(1e-5 + torch.abs(distances_squared))


def MMD_statistic(samples1, samples2, ret_matrix=False):
    """compute mmd with rbf kernel"""
    n_1 = samples1.shape[0]
    n_2 = samples2.shape[0]
    a00 = 1. / (n_1 * (n_1 - 1.))
    a11 = 1. / (n_2 * (n_2 - 1.))
    a01 = - 1. / (n_1 * n_2)

    sample_12 = torch.cat((samples1, samples2), 0)
    distances = l2norm_dist(sample_12, sample_12)

    dist = torch.triu(distances).view(-1)
    gamma = 1 / torch.median(dist[dist > 0.]) ** 2

    kernels = torch.exp(- gamma * distances ** 2)
    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * k_12.sum() +
           a00 * (k_1.sum() - torch.trace(k_1)) +
           a11 * (k_2.sum() - torch.trace(k_2)))
    if ret_matrix:
        return mmd, kernels
    else:
        return mmd


def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic."""
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
           1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
           2.0 / (m * n) * Kxy.sum()


def MMD_bootstrap(null_samples, alter_samples, num_bootstrap):
    n1 = null_samples.shape[0]
    n2 = alter_samples.shape[0]
    _, K = MMD_statistic(null_samples, alter_samples, ret_matrix=True)
    bootstrap_null_samples = torch.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        idx = torch.randperm(n1 + n2)
        K_i = K[idx, idx[:, None]]
        bootstrap_null_samples[i] = MMD2u(K_i, n1, n2)
    return bootstrap_null_samples


def MMD_test(null_samples, alter_samples, bootstrap_null_samples):
    n1 = null_samples.shape[0]
    n2 = alter_samples.shape[0]
    test_statistic = MMD_statistic(null_samples, alter_samples)
    pvalue = torch.mean((bootstrap_null_samples > test_statistic).float())
    return test_statistic, pvalue


def Distance_test_batch(test_name, null_samples, alter_samples, null_model, num_bootstrap=1000, alpha=0.05):
    num_tests = alter_samples.shape[0]
    num_samples = alter_samples.shape[1]
    if test_name == 'KSD-U':
        bootstrap_null_samples = multinomial_bootstrap(test_name, num_bootstrap, num_samples, null_samples,
                                                       null_model.score)
    if test_name == 'MMD':
        bootstrap_null_samples = MMD_bootstrap(null_samples, alter_samples[0], num_bootstrap)

    alter_statisitics = []
    pvalues = []
    start_time = time.time()
    for i in range(num_tests):
        if test_name == 'KSD-U':
            test_statistic, pvalue = KSD_U_test(alter_samples[i], bootstrap_null_samples, null_model.score)
        if test_name == 'MMD':
            test_statistic, pvalue = MMD_test(null_samples, alter_samples[i], bootstrap_null_samples)
        alter_statisitics.append(test_statistic.item())
        pvalues.append(pvalue.item())
    mean_time = (time.time() - start_time) / num_tests
    result_batch = {
        'critical value': torch.quantile(bootstrap_null_samples, 1 - alpha),
        'statisitics': alter_statisitics,
        'pvalues': pvalues,
        'power': sum([pvalue < alpha for pvalue in pvalues]) / num_tests,
        'time': mean_time
    }
    return result_batch


if __name__ == '__main__':
    Z = torch.tensor(np.arange(1, 3001))
    Z = Z.view(3, 1000)


    def score_function(x):
        return -x


    KSD_items, KSD = KSD_statistics(Z.t(), score_function, width='median')
    print(KSD_items.shape, KSD)
