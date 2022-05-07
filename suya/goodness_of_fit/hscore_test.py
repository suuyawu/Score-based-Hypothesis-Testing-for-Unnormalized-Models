"""implementation of hscore test
"""
import torch
import time

def Hscore(samples, null_hscore, alter_hscore):
    """Calculate Hyvarinen Score Difference"""
    Hscore_items = -alter_hscore(samples) + null_hscore(samples)
    #To calculate the scalar for chi2 distribution approximation under the null
    #Hscore_items = Hscore_items*scalar
    return Hscore_items, torch.sum(Hscore_items,-1)

def LRT(samples, null_pdf, alter_pdf):
    """Calculate Likelihood Ratio"""
    LRT_items = 2*(torch.log(alter_pdf(samples)) - torch.log(null_pdf(samples)))
    return LRT_items, torch.sum(LRT_items,-1)

def m_out_n_bootstrap(test_name, num_bootstrap, num_samples, sub_samples, null_samples, null_model, alter_model):
    """Bootstrap algorithm (m out of n) for hypothesis testing by Bickel & Ren (2001)"""
    if test_name == 'HScore':
        null_items, test = Hscore(null_samples, null_model.hscore, alter_model.hscore)
    if test_name == 'LikelihoodRatio Chi2' or  test_name == 'LikelihoodRatio Bootstrap':
        null_items, test = LRT(null_samples, null_model.pdf, alter_model.pdf)
    _index = torch.multinomial(torch.ones(num_samples).repeat(num_bootstrap, 1)/num_samples, sub_samples, replacement=True)
    bootstrap_null_items = torch.gather(null_items.repeat(num_bootstrap, 1), 1, _index)
    bootstrap_null_samples = torch.sum(bootstrap_null_items, dim=-1)
    return bootstrap_null_samples

def Density_test(test_name, alter_samples, bootstrap_null_samples, null_model, alter_model, bootstrap_approx):
    if test_name == 'HScore':
        _, test_statistic = Hscore(alter_samples, null_model.hscore, alter_model.hscore)
    elif test_name == 'LikelihoodRatio Chi2' or test_name == 'LikelihoodRatio Bootstrap':
        _, test_statistic = LRT(alter_samples, null_model.pdf, alter_model.pdf)
    if bootstrap_approx:
        pvalue = torch.mean((bootstrap_null_samples >= test_statistic).float())
    else:
        from scipy.stats import chi2 
        df = 1
        pvalue = 1 - chi2(df).cdf(test_statistic) # since Λ follows χ2
    return test_statistic, pvalue

def Density_test_batch(test_name, null_samples, alter_samples, null_model, alter_model, bootstrap_approx=True, num_bootstrap=1000, alpha=0.05):
    """Perform HScore test or Likelihood Ratio test

    Args:
        test_name (str): 'HScore' or 'LikelihoodRatio'
        null_samples (Tensor in shape (num_samples,)): samples from the null distribution, note that num_samples >= sample_size because we do m out of n bootstrap
        alter_samples (Tensor in shape (num_tests, sample_size)): samples from the alternative distribution
        null_model (object): null distribution model
        alter_model (object): alternative distribution model
        num_bootstrap (int): number of bootstrap resampling
        alpha (float): significance level
    Return:

    """
    num_tests = alter_samples.shape[0]
    sub_samples = alter_samples.shape[1]
    num_samples = null_samples.shape[0]

    bootstrap_null_samples = m_out_n_bootstrap(test_name, num_bootstrap, num_samples, sub_samples, null_samples, null_model, alter_model)
    alter_statisitics = []
    pvalues = []
    start_time = time.time()
    for i in range(num_tests):
        test_statistic, pvalue = Density_test(test_name, alter_samples[i], bootstrap_null_samples, null_model, alter_model, bootstrap_approx)
        alter_statisitics.append(test_statistic.item())
        pvalues.append(pvalue.item())
    mean_time = (time.time() - start_time)/num_tests
    result_batch = {
        'critical value': torch.quantile(bootstrap_null_samples, 1-alpha),
        'statisitics': alter_statisitics,
        'pvalues': pvalues,
        'power':sum([pvalue < alpha for pvalue in pvalues])/num_tests,
        'time': mean_time
    }
    return result_batch

        