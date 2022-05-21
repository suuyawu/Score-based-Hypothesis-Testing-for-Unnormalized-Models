import copy
import numpy as np
import torch
import models
from scipy.stats import chi2
from config import cfg


class HST:
    def __init__(self, num_bootstrap, bootstrap_approx):
        super().__init__()
        self.num_bootstrap = num_bootstrap
        self.bootstrap_approx = bootstrap_approx

    def test(self, null_samples, alter_samples, null_model, alter_model=None):
        num_tests = alter_samples.size(0)
        num_samples_alter = alter_samples.size(1)
        statistic = []
        pvalue = []
        for i in range(num_tests):
            if alter_model is None:
                null_model_emp = eval('models.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                null_model_emp.fit(null_samples[i])
                alter_model = eval('models.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                alter_model.fit(alter_samples[i])
            else:
                null_model_emp = alter_model
            with torch.no_grad():
                bootstrap_null_samples = self.m_out_n_bootstrap(null_samples, num_samples_alter, null_model,
                                                                null_model_emp)
                statistic_i, pvalue_i = self.density_test(alter_samples[i], bootstrap_null_samples, null_model,
                                                          alter_model, self.bootstrap_approx)
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue

    def m_out_n_bootstrap(self, null_samples, num_samples_alter, null_model, alter_model):
        """Bootstrap algorithm (m out of n) for hypothesis testing by Bickel & Ren (2001)"""
        null_samples = null_samples.view(-1, *null_samples.size()[2:])
        num_samples_null = null_samples.size(0)
        null_items, _ = self.hst(null_samples, null_model.hscore, alter_model.hscore)
        _index = torch.multinomial(
            null_items.new_ones(num_samples_null).repeat(self.num_bootstrap, 1) / num_samples_null, num_samples_alter,
            replacement=True)
        null_items = null_items.repeat(self.num_bootstrap, 1)
        bootstrap_null_items = torch.gather(null_items, 1, _index)
        bootstrap_null_samples = torch.sum(bootstrap_null_items, dim=-1)
        return bootstrap_null_samples

    def multinomial_bootstrap(self, null_samples, null_model, alter_model):
        """Bootstrap algorithm for U-statistics by Huskova & Janssen (1993)"""
        num_samples_alter = null_samples.size(0)
        null_items, _ = self.hst(null_samples, null_model.hscore, alter_model.hscore)
        weights_exp1, weights_exp2 = self.multinomial_weights(num_samples_alter)
        weights_exp1, weights_exp2 = weights_exp1.to(null_samples.device), weights_exp2.to(null_samples.device)
        null_items = torch.unsqueeze(null_items, dim=0)  # 1 x N x N
        bootstrap_null_samples = (weights_exp1 - 1. / num_samples_alter) * null_items * (
                weights_exp2 - 1. / num_samples_alter)  # m x N x N
        bootstrap_null_samples = torch.sum(torch.sum(bootstrap_null_samples, dim=-1), dim=-1)
        return bootstrap_null_samples

    def multinomial_weights(self, num_samples):
        """Sample multinomial weights for bootstrap by Huskova & Janssen (1993)"""
        weights = np.random.multinomial(num_samples, np.ones(num_samples) / num_samples, size=self.num_bootstrap)
        weights = weights / num_samples
        weights = torch.from_numpy(weights)
        weights_exp1 = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x N
        return weights_exp1, weights_exp2

    def hst(self, samples, null_hscore, alter_hscore):
        """Calculate Hyvarinen Score Difference"""
        Hscore_items = -alter_hscore(samples) + null_hscore(samples)
        # To calculate the scalar for chi2 distribution approximation under the null
        # Hscore_items = Hscore_items*scalar
        Hscore_items = Hscore_items.reshape(-1)
        test_statistic = torch.sum(Hscore_items, -1)
        return Hscore_items, test_statistic

    def density_test(self, alter_samples, bootstrap_null_samples, null_model, alter_model, bootstrap_approx):
        _, test_statistic = self.hst(alter_samples, null_model.hscore, alter_model.hscore)
        test_statistic = test_statistic.item()
        if bootstrap_approx:
            pvalue = torch.mean((bootstrap_null_samples >= test_statistic).float()).item()
        else:
            df = 1
            pvalue = 1 - chi2(df).cdf(test_statistic)  # since Λ follows χ2
        return test_statistic, pvalue
