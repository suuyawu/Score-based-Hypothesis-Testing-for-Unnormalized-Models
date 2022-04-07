"""
Test goodness of fit by Hyvarinen score-based test statistic, compared with likelihood ratio test.
We consider the hypothesis testing, 
    H_0: theta = theta_0 v.s. theta \neq theta_0
To do simulations on 1d Gaussian, theta = (mu, sigma)
"""
from pyexpat import model
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from distributions import *

def ks_test(XX1, model0):
    """
    Kolmogorov-Smirnov test for goodness of fit
    """
    pvalue_ks = np.array([stats.kstest(X1, model0.cdf, alternative='two-sided', mode='auto')[1] for X1 in XX1])
    power_ks = ((pvalue_ks < 0.05).sum() / len(pvalue_ks))
    return power_ks

def cm_test(XX1, model0):
    """
    Cramér-von Mises test for goodness of fit
    """
    pvalue_cm = np.array([stats.cramervonmises(X1, model0.cdf).pvalue for X1 in XX1])
    power_cm = ((pvalue_cm < 0.05).sum() / len(pvalue_cm))
    return power_cm

def st_stat(X, dist_name, params1, params0):
    """
    calculate score-based test statistics
    """
    if dist_name == '1dgmm':
        model1 = OneDimensionGM(params1['omega'], params1['mean'], params1['var'])
        model0 = OneDimensionGM(params0['omega'], params0['mean'], params0['var'])
    if dist_name == '1dnorm':
        model1 = OneDimensionNormal(params1['mean'], params1['std'])
        model0 = OneDimensionNormal(params0['mean'], params0['std'])
    st_X = [-model1.hscore(x) + model0.hscore(x) for x in X]
    return sum(st_X)

def lrt_stat(X, dist_name, params1, params0):
    """
    calculate lrt test statistics
    """
    if dist_name == '1dgmm':
        model1 = OneDimensionGM(params1['omega'], params1['mean'], params1['var'])
        model0 = OneDimensionGM(params0['omega'], params0['mean'], params0['var'])
    if dist_name == '1dnorm':
        model1 = OneDimensionNormal(params1['mean'], params1['std'])
        model0 = OneDimensionNormal(params0['mean'], params0['std'])
    lrt_X = [2*(np.log(model1.pdf(x, density=True)) - np.log(model0.pdf(x, density=True))) for x in X]
    return sum(lrt_X)

def power_function(XX1, XX0, dist_name, params0, params1):
    """
    calculate the emprical power
    """
    st_X_1 = np.array([st_stat(X1, dist_name, params1, params0) for X1 in XX1])
    lrt_X_1 = np.array([lrt_stat(X1, dist_name, params1, params0) for X1 in XX1])
    st_X_0 = np.array([st_stat(X0, dist_name, params1, params0) for X0 in XX0])
    lrt_X_0 = np.array([lrt_stat(X0, dist_name, params1, params0) for X0 in XX0])

    tau_st = np.quantile(st_X_0, 0.95)
    tau_lrt = np.quantile(lrt_X_0, 0.95)

    power_st = ((st_X_1 > tau_st).sum() / len(st_X_1))
    power_lrt = ((lrt_X_1 > tau_lrt).sum() / len(lrt_X_1))
    size_st = ((st_X_0 > tau_st).sum() / len(st_X_0))
    size_lrt = ((lrt_X_0 > tau_lrt).sum() / len(lrt_X_0))
    return power_st, power_lrt, size_st, size_lrt 

def simulation(m, eval, dist_name, test_names, params, perturb='mean'):
    if eval == 'perturbation':
        xaxis = epss = np.array([0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.5])
        ns = np.array([100]*len(epss))
        noise_std = np.array([0]*len(epss))
    if eval == 'sample_size':
        xaxis = ns = np.array([10, 20, 30, 40, 50, 70, 100, 200])
        epss = np.array([0.8]*len(ns))
        noise_std = np.array([0]*len(ns))
    if eval == 'noisy_data':
        xaxis = noise_std = np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1])
        epss = np.array([0.8]*len(noise_std))
        ns = np.array([100]*len(noise_std))
    powers = {}
    for name in test_names:
        powers.update({name:[]})
    for (n, eps, std_) in zip(ns, epss, noise_std):
        print(n, eps)
        if dist_name == '1dnorm':
            model0 = OneDimensionNormal(params['mean'], params['std'])
            params_perturbed = params.copy()
            params_perturbed['mean'] = params_perturbed['mean'] + eps
            model1 = OneDimensionNormal(params_perturbed['mean'], params_perturbed['std'])
        elif dist_name == '1dgmm':
            model0 = OneDimensionGM(params['omega'], params['mean'], params['var'])
            params_perturbed = params.copy()
            if perturb == 'mean':
                params_perturbed['mean'] = params_perturbed['mean'] +eps
            elif perturb == 'omega':
                params_perturbed['omega'] = np.exp(np.log(params_perturbed['omega']) + eps)
                params_perturbed['omega'][1] = 1- params_perturbed['omega'][0]
            elif perturb == 'var':
                params_perturbed['var'] = np.exp(np.log(params_perturbed['var']) + eps)
            model1 = OneDimensionGM(params_perturbed['omega'], params_perturbed['mean'], params_perturbed['var'])
        XX0 = model0.sample(m, n)
        XX1 = model1.sample(m, n)+np.random.randn(m, n)*std_
        power_st, power_lrt, _, _ = power_function(XX1, XX0, dist_name, params, params_perturbed)
        powers['HScore'].append(power_st)
        powers['Likelihood Ratio'].append(power_lrt)
        if 'Kolmogorov Smirnov' in test_names:
            powers['Kolmogorov Smirnov'].append(ks_test(XX1, model0))
        if 'Cramér-von Mises' in test_names:
            powers['Cramér-von Mises'].append(cm_test(XX1, model0))
    
    chars = ['-rD', '-go', '-bx', '-mp']
    for i, name in enumerate(test_names):
        plt.plot(xaxis, np.array(powers[name]), chars[i], label = '{} test'.format(name))
    plt.legend()
    plt.title('Power comparision with fixed test size 0.05')
    plt.savefig('./output/{}/{}_{}_power.pdf'.format(eval, dist_name, perturb))
    plt.close()

if __name__ == '__main__':
    params = {
        'omega':np.array([0.2, 0.8]),
        'mean':np.array([0, 5]),
        'var':np.array([1, 1])
    }
    perturbs = ['mean', 'var', 'omega']
    for perturb in perturbs:
        simulation(m=1000, eval='noisy_data', dist_name = '1dgmm', test_names=['HScore', 'Likelihood Ratio', 'Kolmogorov Smirnov', 'Cramér-von Mises'], params = params, perturb=perturb)