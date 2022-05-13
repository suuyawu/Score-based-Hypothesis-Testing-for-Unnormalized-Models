import numpy as np
import torch


class HST:
    def __init__(self, num_bootstrap, bootstrap_approx):
        super().__init__()
        self.num_bootstrap = num_bootstrap
        self.bootstrap_approx = bootstrap_approx

    def test(self, null_samples, alter_samples, null_model, alter_model=None):
        if alter_model is None:
            pass
            # need model fitting
        bootstrap_null_samples = self.m_out_n_bootstrap(len(null_samples), len(alter_samples), null_samples, null_model,
                                                        alter_model)
        statistic = []
        pvalue = []
        for i in range(len(alter_samples)):
            statistic_i, pvalue_i = self.density_test(alter_samples[i], bootstrap_null_samples, null_model,
                                                      alter_model, self.bootstrap_approx)
            statistic.append(statistic_i)
            pvalue.append(pvalue_i)
        return statistic, pvalue

    def m_out_n_bootstrap(self, num_samples_null, num_samples_alter, null_samples, null_model,
                          alter_model):
        """Bootstrap algorithm (m out of n) for hypothesis testing by Bickel & Ren (2001)"""
        null_items, test = self.hst(null_samples, null_model.hscore, alter_model.hscore)
        _index = torch.multinomial(torch.ones(num_samples_null).repeat(self.num_bootstrap, 1) / num_samples_null,
                                   num_samples_alter, replacement=True)
        bootstrap_null_items = torch.gather(null_items.repeat(self.num_bootstrap, 1), 1, _index)
        bootstrap_null_samples = torch.sum(bootstrap_null_items, dim=-1)
        return bootstrap_null_samples

    def hst(self, samples, null_hscore, alter_hscore):
        """Calculate Hyvarinen Score Difference"""
        Hscore_items = -alter_hscore(samples) + null_hscore(samples)
        # To calculate the scalar for chi2 distribution approximation under the null
        # Hscore_items = Hscore_items*scalar
        return Hscore_items, torch.sum(Hscore_items, -1)

    def density_test(self, alter_samples, bootstrap_null_samples, null_model, alter_model, bootstrap_approx):
        _, test_statistic = self.hst(alter_samples, null_model.hscore, alter_model.hscore)
        test_statistic = test_statistic.item()
        if bootstrap_approx:
            pvalue = torch.mean((bootstrap_null_samples > test_statistic).float()).item()
        else:
            from scipy.stats import chi2
            df = 1
            pvalue = 1 - chi2(df).cdf(test_statistic)  # since Λ follows χ2
        return test_statistic, pvalue
