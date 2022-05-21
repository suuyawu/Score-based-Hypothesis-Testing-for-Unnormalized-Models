import numpy as np
import torch


class MMD:
    def __init__(self, num_bootstrap):
        super().__init__()
        self.num_bootstrap = num_bootstrap

    def test(self, null_samples, alter_samples):
        with torch.no_grad():
            statistic = []
            pvalue = []
            num_tests = alter_samples.size(0)
            for i in range(num_tests):
                kernel_hyper = self.make_kernel_hyper(null_samples[i], alter_samples[i])
                statistic_i, bootstrap_null_samples = self.MMD_bootstrap(null_samples[i], alter_samples[i],
                                                                         kernel_hyper)
                pvalue_i = self.MMD_test(statistic_i, bootstrap_null_samples)
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue

    def make_kernel_hyper(self, null_samples, alter_samples):
        all_samples = torch.cat((null_samples, alter_samples), dim=0)
        median_dist = self.median_heruistic(all_samples, all_samples.clone())
        bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
        kernel_hyper = {'bandwidth': bandwidth}
        return kernel_hyper

    def median_heruistic(self, sample1, sample2):
        with torch.no_grad():
            G = torch.sum(sample1 * sample1, dim=-1)  # N or * x N
            G_exp = torch.unsqueeze(G, dim=-2)  # 1 x N or * x1 x N
            H = torch.sum(sample2 * sample2, dim=-1)
            H_exp = torch.unsqueeze(H, dim=-1)  # N x 1 or * * x N x 1
            dist = G_exp + H_exp - 2 * sample2.matmul(torch.transpose(sample1, -1, -2))  # N x N or  * x N x N
            if len(dist.shape) == 3:
                dist = dist[torch.triu(torch.ones(dist.shape)) == 1].view(dist.shape[0], -1)  # * x (NN)
                median_dist, _ = torch.median(dist, dim=-1)  # *
            else:
                dist = (dist - torch.tril(dist)).view(-1)
                median_dist = torch.median(dist[dist > 0.])
        return median_dist.clone().detach()

    def MMD_bootstrap(self, null_samples, alter_samples, kernel_hyper):
        num_samples = null_samples.shape[0]
        weights = np.random.multinomial(num_samples, np.ones(num_samples) / num_samples, size=self.num_bootstrap)
        weights = weights / num_samples
        weights = torch.from_numpy(weights).type(alter_samples.type())
        MMD, MMD_comp = self.MMD_statistic(null_samples, alter_samples, self.SE_kernel_multi, kernel_hyper, flag_U=True,
                                           flag_simple_U=True)
        MMD = MMD.item()
        # Now compute bootstrap samples
        with torch.no_grad():
            MMD_comp = torch.unsqueeze(MMD_comp, dim=0)  # 1 x N x N
            weights_exp = torch.unsqueeze(weights, dim=-1)  # m x N x 1
            weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x n
            bootstrap_samples = (weights_exp - 1. / num_samples) * MMD_comp * (
                    weights_exp2 - 1. / num_samples)  # m x N x N
            bootstrap_samples = torch.sum(torch.sum(bootstrap_samples, dim=-1), dim=-1)
        return MMD, bootstrap_samples

    def SE_kernel_multi(self, sample1, sample2, **kwargs):
        '''
        Compute the multidim square exponential kernel
        :param sample1: x : N x N x dim
        :param sample2: y : N x N x dim
        :param kwargs: kernel hyper-parameter:bandwidth
        :return:
        '''
        bandwidth = kwargs['kernel_hyper']['bandwidth']
        if len(sample1.shape) == 4:  # * x N x d
            bandwidth = bandwidth.unsqueeze(-1).unsqueeze(-1)
        sample_diff = sample1 - sample2  # N x N x dim
        norm_sample = torch.norm(sample_diff, dim=-1) ** 2  # N x N or * x N x N
        K = torch.exp(-norm_sample / (bandwidth ** 2 + 1e-9))
        return K

    def MMD_statistic(self, samples1, samples2, kernel, kernel_hyper, flag_U=True, flag_simple_U=True):
        # samples1: N x dim
        # samples2: N x dim
        n = samples1.shape[0]
        m = samples2.shape[0]
        if m != n and flag_simple_U:
            raise ValueError('If m is not equal to n, flag_simple_U must be False')
        samples1_exp1 = torch.unsqueeze(samples1, dim=1)  # N x 1 x dim
        samples1_exp2 = torch.unsqueeze(samples1, dim=0)  # 1 x N x dim
        samples2_exp1 = torch.unsqueeze(samples2, dim=1)  # N x 1 x dim
        samples2_exp2 = torch.unsqueeze(samples2, dim=0)  # 1 x N x dim
        # Term1
        K1 = kernel(samples1_exp1, samples1_exp2, kernel_hyper=kernel_hyper)  # N x N
        if flag_U:
            K1 = K1 - torch.diag(torch.diag(K1))
        # Term3
        K3 = kernel(samples2_exp1, samples2_exp2, kernel_hyper=kernel_hyper)  # N x N
        if flag_U:
            K3 = K3 - torch.diag(torch.diag(K3))
        # Term2
        if flag_simple_U:
            K2_comp = kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper)
            K2_comp = K2_comp - torch.diag(torch.diag(K2_comp))
            K2 = K2_comp + K2_comp.t()
        else:
            K2 = 2 * kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper)  # N x N
        if flag_U:
            if flag_simple_U:
                MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * (m - 1)) * torch.sum(K2)

            else:
                MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * n) * torch.sum(K2)
        else:
            MMD = torch.sum(K1 + K3 - K2) / (m * n)
        return MMD, K1 + K3 - K2

    def MMD_test(self, test_statistic, bootstrap_null_samples):
        pvalue = torch.mean((bootstrap_null_samples >= test_statistic).float()).item()
        return pvalue
