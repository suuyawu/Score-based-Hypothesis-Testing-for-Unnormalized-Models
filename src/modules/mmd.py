import numpy as np
import torch
from .permutation_test import permutation_test_mat


class MMD:
    def __init__(self, num_permutations):
        super().__init__()
        self.num_permutations = num_permutations

    def test(self, null_samples, alter_samples):
        with torch.no_grad():
            # bootstrap_null_samples = self.MMD_bootstrap(null_samples, alter_samples, self.num_bootstrap)
            statistic = []
            pvalue = []
            for i in range(len(null_samples)):
                n_1 = null_samples[i].size(0)
                n_2 = alter_samples[i].size(0)
                self.n_1 = n_1
                self.n_2 = n_2
                self.a00 = 1. / (n_1 * (n_1 - 1))
                self.a11 = 1. / (n_2 * (n_2 - 1))
                self.a01 = - 1. / (n_1 * n_2)
                statistic_i, mat_i = self.MMD_statistic(null_samples[i], alter_samples[i], ret_matrix=True)
                pvalue_i = self.pval(mat_i, n_permutations=self.num_permutations)
                statistic.append(statistic_i.item())
                pvalue.append(pvalue_i)
        return statistic, pvalue

    def MMD_statistic(self, sample_1, sample_2, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = self.pdist(sample_12, sample_12, norm=2)

        dist = torch.triu(distances).view(-1)
        gamma = 1 / torch.median(dist[dist > 0.]) ** 2
        alphas = [gamma]

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(- alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[:self.n_1, :self.n_1]
        k_2 = kernels[self.n_1:, self.n_1:]
        k_12 = kernels[:self.n_1, self.n_1:]

        mmd = (2 * self.a01 * k_12.sum() +
               self.a00 * (k_1.sum() - torch.trace(k_1)) +
               self.a11 * (k_2.sum() - torch.trace(k_2)))
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.
        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.
        Returns
        -------
        float
            The estimated p-value."""
        return permutation_test_mat(distances.cpu().numpy(),
                                    self.n_1, self.n_2,
                                    n_permutations,
                                    a00=self.a00, a11=self.a11, a01=self.a01)

    def pdist(self, sample_1, sample_2, norm=2, eps=1e-5):
        r"""Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.:
            norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
            norms = (norms_1.expand(n_1, n_2) +
                     norms_2.transpose(0, 1).expand(n_1, n_2))
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1. / norm)

    #
    #
    #
    # def MMD_bootstrap(self, null_samples, alter_samples, num_bootstrap):
    #     n1 = null_samples.shape[0]
    #     n2 = alter_samples.shape[0]
    #     _, K = self.MMD_statistic(null_samples, alter_samples, ret_matrix=True)
    #     bootstrap_null_samples = torch.zeros(num_bootstrap)
    #     for i in range(num_bootstrap):
    #         idx = torch.randperm(n1 + n2)
    #         K_i = K[idx, idx[:, None]]
    #         bootstrap_null_samples[i] = self.MMD2u(K_i, n1, n2)
    #     return bootstrap_null_samples
    #
    # def MMD_test(self, null_samples, alter_samples, bootstrap_null_samples):
    #     n1 = null_samples.shape[0]
    #     n2 = alter_samples.shape[0]
    #     test_statistic = self.MMD_statistic(null_samples, alter_samples)
    #     pvalue = torch.mean((bootstrap_null_samples > test_statistic).float())
    #     return test_statistic, pvalue
    #
    # def MMD_statistic(self, samples1, samples2, ret_matrix=False):
    #     """compute mmd with rbf kernel"""
    #     n_1 = samples1.shape[0]
    #     n_2 = samples2.shape[0]
    #     a00 = 1. / (n_1 * (n_1 - 1.))
    #     a11 = 1. / (n_2 * (n_2 - 1.))
    #     a01 = - 1. / (n_1 * n_2)
    #
    #     sample_12 = torch.cat((samples1, samples2), 0)
    #     distances = self.l2norm_dist(sample_12, sample_12)
    #
    #     dist = torch.triu(distances).view(-1)
    #     gamma = 1 / torch.median(dist[dist > 0.]) ** 2
    #
    #     kernels = torch.exp(- gamma * distances ** 2)
    #     k_1 = kernels[:n_1, :n_1]
    #     k_2 = kernels[n_1:, n_1:]
    #     k_12 = kernels[:n_1, n_1:]
    #
    #     mmd = (2 * a01 * k_12.sum() +
    #            a00 * (k_1.sum() - torch.trace(k_1)) +
    #            a11 * (k_2.sum() - torch.trace(k_2)))
    #     if ret_matrix:
    #         return mmd, kernels
    #     else:
    #         return mmd
    #
    # def l2norm_dist(self, sample_1, sample_2):
    #     """Compute the matrix of all squared pairwise distances.
    #
    #     Args:
    #         sample_1 (Tensor in shape (n_1, d)): The first sample
    #         sample_2 (Tensor in shape (n_2, d)): The second sample
    #
    #     Returns
    #         Tensor in shape (n_1, n_2): The [i, j]-th entry is equal to ``|| sample_1[i, :] - sample_2[j, :] ||_p``.
    #     """
    #     n_1, n_2 = sample_1.size(0), sample_2.size(0)
    #     norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
    #     norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
    #     norms = (norms_1.expand(n_1, n_2) +
    #              norms_2.transpose(0, 1).expand(n_1, n_2))
    #     distances_squared = norms - 2 * sample_1.mm(sample_2.t())
    #     return torch.sqrt(1e-5 + torch.abs(distances_squared))

    # def MMD2u(self, K, m, n):
    #     """The MMD^2_u unbiased statistic."""
    #     Kx = K[:m, :m]
    #     Ky = K[m:, m:]
    #     Kxy = K[:m, m:]
    #     return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
    #            1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
    #            2.0 / (m * n) * Kxy.sum()
