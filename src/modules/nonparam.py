from scipy import stats


class CVM:
    def __init__(self):
        super().__init__()

    def test(self, alter_samples, null_model):
        statistic = []
        pvalue = []
        for i in range(len(alter_samples)):
            cm = stats.cramervonmises(alter_samples[i], null_model.cdf)
            statistic_i = cm.statistic
            pvalue_i = cm.pvalue
            statistic.append(statistic_i)
            pvalue.append(pvalue_i)
        return statistic, pvalue


class KS:
    def __init__(self):
        super().__init__()

    def test(self, input, null_model):
        statistic = []
        pvalue = []
        for i in range(len(input)):
            statistic_i, pvalue_i = stats.kstest(input[i], null_model.cdf, alternative='two-sided', mode='auto')
            statistic.append(statistic_i)
            pvalue.append(pvalue_i)
        return statistic, pvalue
