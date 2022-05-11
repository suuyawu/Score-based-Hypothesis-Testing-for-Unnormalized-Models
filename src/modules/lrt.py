import numpy as np
import torch


class LRT:
    def __init__(self):
        super().__init__()
        self.num_bootstrap = 1000
        self.bootstrap_approx = True

    def test(self, null_samples, alter_samples, null_model, alter_model=None):
        if alter_model is None:
            pass
        #     need model fitting
        bootstrap_null_samples = self.m_out_n_bootstrap(self.num_bootstrap, len(null_samples), len(alter_samples),
                                                        null_samples, null_model, alter_model)
        statistic = []
        pvalue = []
        for i in range(len(alter_samples)):
            statistic_i, pvalue_i = self.density_test(alter_samples[i], bootstrap_null_samples, null_model,
                                                      alter_model, self.bootstrap_approx)
            statistic.append(statistic_i)
            pvalue.append(pvalue_i)
        return statistic, pvalue

    def m_out_n_bootstrap(self, num_bootstrap, num_samples, sub_samples, null_samples, null_model,
                          alter_model):
        """Bootstrap algorithm (m out of n) for hypothesis testing by Bickel & Ren (2001)"""
        null_items, test = self.lrt(null_samples, null_model.pdf, alter_model.pdf)
        _index = torch.multinomial(torch.ones(num_samples).repeat(num_bootstrap, 1) / num_samples, sub_samples,
                                   replacement=True)
        bootstrap_null_items = torch.gather(null_items.repeat(num_bootstrap, 1), 1, _index)
        bootstrap_null_samples = torch.sum(bootstrap_null_items, dim=-1)
        return bootstrap_null_samples

    def density_test(self, alter_samples, bootstrap_null_samples, null_model, alter_model, bootstrap_approx):
        _, test_statistic = self.lrt(alter_samples, null_model.pdf, alter_model.pdf)
        if bootstrap_approx:
            pvalue = torch.mean((bootstrap_null_samples > test_statistic).float())
        else:
            from scipy.stats import chi2
            df = 1
            pvalue = 1 - chi2(df).cdf(test_statistic)  # since Λ follows χ2
        return test_statistic, pvalue

    def lrt(self, samples, null_pdf, alter_pdf):
        """Calculate Likelihood Ratio"""
        LRT_items = 2 * (torch.log(alter_pdf(samples)) - torch.log(null_pdf(samples)))
        return LRT_items, torch.sum(LRT_items, -1)
