import torch
from scipy import stats


class CVM:
    def __init__(self):
        super().__init__()

    def test(self, alter_samples, null_model):
        with torch.no_grad():
            statistic = []
            pvalue = []
            for i in range(len(alter_samples)):
                cm = stats.cramervonmises(alter_samples[i].reshape(-1), null_model.cdf_numpy)
                statistic_i = cm.statistic
                pvalue_i = cm.pvalue
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue

class KS:
    def __init__(self):
        super().__init__()

    def test(self, alter_samples, null_model):
        with torch.no_grad():
            statistic = []
            pvalue = []
            for i in range(len(alter_samples)):
                statistic_i, pvalue_i = stats.kstest(alter_samples[i].reshape(-1), null_model.cdf_numpy,
                                                     alternative='two-sided', mode='auto')
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue
