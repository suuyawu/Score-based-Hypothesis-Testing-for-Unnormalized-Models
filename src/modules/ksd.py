import numpy as np
import torch


class KSD:
    def __init__(self, num_bootstrap, V_stat):
        super().__init__()
        self.num_bootstrap = num_bootstrap
        self.V_stat = V_stat

    def test(self, null_samples, alter_samples, null_model):
        with torch.no_grad():
            statistic = []
            pvalue = []
            num_tests = alter_samples.size(0)
            for i in range(num_tests):
                bootstrap_null_samples = self.multinomial_bootstrap(null_samples[i], null_model)
                statistic_i, pvalue_i = self.KSD_U_test(alter_samples[i], bootstrap_null_samples, null_model)
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue

    def multinomial_bootstrap(self, null_samples, null_model):
        """Bootstrap algorithm for U-statistics by Huskova & Janssen (1993)"""
        num_samples_alter = null_samples.size(0)
        null_items, _ = self.KSD_statistics(null_samples, null_model.score, V_stat=self.V_stat)
        weights_exp1, weights_exp2 = self.multinomial_weights(num_samples_alter)
        weights_exp1, weights_exp2 = weights_exp1.to(null_samples.device), weights_exp2.to(null_samples.device)
        null_items = torch.unsqueeze(null_items, dim=0)  # 1 x N x N
        bootstrap_null_samples = (weights_exp1 - 1. / num_samples_alter) * null_items * (
                weights_exp2 - 1. / num_samples_alter)  # m x N x N
        bootstrap_null_samples = torch.sum(torch.sum(bootstrap_null_samples, dim=-1), dim=-1)
        return bootstrap_null_samples

    def KSD_U_test(self, alter_samples, bootstrap_null_samples, null_model):
        _, test_statistic = self.KSD_statistics(alter_samples, null_model.score, V_stat=self.V_stat)
        test_statistic = test_statistic.item()
        pvalue = (bootstrap_null_samples >= test_statistic).float().mean().item()
        return test_statistic, pvalue

    def multinomial_weights(self, num_samples):
        """Sample multinomial weights for bootstrap by Huskova & Janssen (1993)"""
        weights = np.random.multinomial(num_samples, np.ones(num_samples) / num_samples, size=self.num_bootstrap)
        weights = weights / num_samples
        weights = torch.from_numpy(weights)
        weights_exp1 = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x N
        return weights_exp1, weights_exp2

    def KSD_statistics(self, y, score_func, width='median', V_stat=False):
        """KSD for goodness of fit test pytorch implementation of https://rdrr.io/cran/KSD/"""
        # set up the the bandwidth of RBF Kernel
        if width == 'heuristic':
            h = self.ratio_median_heuristic(y, score_func)
        else:
            h = torch.sqrt(0.5 * self.find_median_distance(y, y))
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

    def find_median_distance(self, Z, T):
        """return the median of the data distance
        Args:
            Z (Tensor in size (num_samples, dim)): the samples
        """
        Q = torch.sum(Z * Z, dim=-1).unsqueeze(0)
        R = torch.sum(T * T, dim=-1).unsqueeze(1)
        dist = Q + R - 2 * torch.matmul(Z, T.t())
        dist = torch.triu(dist).view(-1)
        return torch.median(dist[dist > 0.])

    def ratio_median_heuristic(self, Z, score_func):
        """return the bandwidth of kernel by ratio of median heuristic
        Args:
            Z (Tensor in size (num_samples, dim)): the samples
            score_func (func): true grad_x_p(x)
        """
        G = torch.sum(Z * Z, dim=-1)
        Q = torch.unsqueeze(G, dim=0)
        R = torch.unsqueeze(G, dim=1)
        dist = Q + R - 2 * torch.matmul(Z, Z.t())
        dist = torch.triu(dist).view(-1)
        dist_median = torch.median(dist[dist > 0.])
        _zscore = score_func(Z)
        _zscore_sq = torch.matmul(_zscore, _zscore.t())
        _zscore_sq = torch.triu(_zscore_sq, diagonal=1).reshape(-1)
        _zscore_median = torch.median(_zscore_sq[_zscore_sq > 0.])
        bandwidth = (dist_median / _zscore_median) ** 0.25
        return bandwidth
