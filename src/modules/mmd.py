import numpy as np
import torch


class MMD:
    def __init__(self):
        super().__init__()
        self.num_bootstrap = 1000

    def test(self, null_samples, alter_samples):
        bootstrap_null_samples = self.MMD_bootstrap(null_samples, alter_samples, self.num_bootstrap)
        statistic = []
        pvalue = []
        for i in range(len(alter_samples)):
            statistic_i, pvalue_i = self.MMD_test(null_samples, alter_samples[i], bootstrap_null_samples)
            statistic.append(statistic_i)
            pvalue.append(pvalue_i)
        return statistic, pvalue

    def MMD_bootstrap(self, null_samples, alter_samples, num_bootstrap):
        n1 = null_samples.shape[0]
        n2 = alter_samples.shape[0]
        _, K = self.MMD_statistic(null_samples, alter_samples, ret_matrix=True)
        bootstrap_null_samples = torch.zeros(num_bootstrap)
        for i in range(num_bootstrap):
            idx = torch.randperm(n1 + n2)
            K_i = K[idx, idx[:, None]]
            bootstrap_null_samples[i] = self.MMD2u(K_i, n1, n2)
        return bootstrap_null_samples

    def MMD_test(self, null_samples, alter_samples, bootstrap_null_samples):
        n1 = null_samples.shape[0]
        n2 = alter_samples.shape[0]
        test_statistic = self.MMD_statistic(null_samples, alter_samples)
        pvalue = torch.mean((bootstrap_null_samples > test_statistic).float())
        return test_statistic, pvalue

    def MMD_statistic(self, samples1, samples2, ret_matrix=False):
        """compute mmd with rbf kernel"""
        n_1 = samples1.shape[0]
        n_2 = samples2.shape[0]
        a00 = 1. / (n_1 * (n_1 - 1.))
        a11 = 1. / (n_2 * (n_2 - 1.))
        a01 = - 1. / (n_1 * n_2)

        sample_12 = torch.cat((samples1, samples2), 0)
        distances = self.l2norm_dist(sample_12, sample_12)

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

    def l2norm_dist(self, sample_1, sample_2):
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

    def MMD2u(self, K, m, n):
        """The MMD^2_u unbiased statistic."""
        Kx = K[:m, :m]
        Ky = K[m:, m:]
        Kxy = K[:m, m:]
        return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
               1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
               2.0 / (m * n) * Kxy.sum()
