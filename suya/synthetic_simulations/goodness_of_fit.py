"""
Test goodness of fit by Hyvarinen score-based test statistic, compared with likelihood ratio test.
We consider the hypothesis testing, 
    H_0: theta = theta_0 v.s. theta \neq theta_0
To do simulations on 1d Gaussian, theta = (mu, sigma)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from distributions import *
from ksd_tests import KSD_test_power
from kernel_mmd import kernel_two_sample_test

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

def st_stat(X, model0, model1):
    """
    calculate score-based test statistics
    """
    st_X = [-model1.hscore(x) + model0.hscore(x) for x in X]
    return sum(st_X)

def lrt_stat(X, model0, model1):
    """
    calculate lrt test statistics
    """
    lrt_X = [2*(np.log(model1.pdf(x, density=True)) - np.log(model0.pdf(x, density=True))) for x in X]
    return sum(lrt_X)
def power_function(XX1, XX0, model0, model1):
    """
    calculate the emprical power
    """
    st_X_1 = np.array([st_stat(X1, model0, model1) for X1 in XX1])
    lrt_X_1 = np.array([lrt_stat(X1, model0, model1) for X1 in XX1])
    st_X_0 = np.array([st_stat(X0, model0, model1) for X0 in XX0])
    lrt_X_0 = np.array([lrt_stat(X0, model0, model1) for X0 in XX0])

    tau_st = np.quantile(st_X_0, 0.95)
    tau_lrt = np.quantile(lrt_X_0, 0.95)

    power_st = ((st_X_1 > tau_st).sum() / len(st_X_1))
    power_lrt = ((lrt_X_1 > tau_lrt).sum() / len(lrt_X_1))
    size_st = ((st_X_0 > tau_st).sum() / len(st_X_0))
    size_lrt = ((lrt_X_0 > tau_lrt).sum() / len(lrt_X_0))
    return power_st, power_lrt, size_st, size_lrt 

def simulation(m, eval, dist_name, test_names, params, perturb='mean'):
    if eval == 'perturbation':
        xaxis = epss = np.array([0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2])
        ns = np.array([100]*len(epss))
        noise_std = np.array([0]*len(epss))
    if eval == 'sample_size':
        xaxis = ns = np.array([10, 20, 30, 40, 50, 70, 100, 200])
        epss = np.array([0.8]*len(ns))
        noise_std = np.array([0]*len(ns))
    if eval == 'noisy_data':
        xaxis = noise_std = np.array([0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5, 0.8, 1])
        epss = np.array([0.8]*len(noise_std))
        ns = np.array([100]*len(noise_std))
    powers = {}
    for name in test_names:
        powers.update({name:[]})
    for (n, eps, std_) in zip(ns, epss, noise_std):
        print(n, eps, std_)
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
        elif dist_name == 'gb_rbm':
            model0 = GB_RBM(params['W'], params['b'], params['c'])
            params_perturbed = params.copy()
            if perturb == 'W':
                params_perturbed['W'] = params_perturbed['W'] +eps
            elif perturb == 'b':
                params_perturbed['b'] = params_perturbed['b'] +eps
            elif perturb == 'c':
                params_perturbed['c'] = params_perturbed['c'] +eps
            model1 = GB_RBM(params_perturbed['W'], params_perturbed['b'], params_perturbed['c'])
        XX0 = model0.sample(m, n)
        XX1 = model1.sample(m, n)+np.random.randn(*XX0.shape)*std_
        power_st, power_lrt, _, _ = power_function(XX1, XX0, model0, model1)
        powers['HScore'].append(power_st)
        powers['Likelihood Ratio'].append(power_lrt)
        if 'Kolmogorov Smirnov' in test_names:
            powers['Kolmogorov Smirnov'].append(ks_test(XX1, model0))
        if 'Cramér-von Mises' in test_names:
            powers['Cramér-von Mises'].append(cm_test(XX1, model0))
        if 'KSD-U' in test_names:
            powers['KSD-U'].append(KSD_test_power(XX1, model0.score_function, width='median', nboot=1000))
        if 'Kernel-MMD' in test_names:
            for X0, X1 in zip(XX0, XX1):
                _, _, pvalue = kernel_two_sample_test(X0, X1, kernel_function='rbf', iterations=1000)
            powers['Kernel-MMD'].append(kmmd_power)
    chars = ['-rD', '-go', '-bx', '-mp']
    for i, name in enumerate(test_names):
        plt.plot(xaxis, np.array(powers[name]), chars[i], label = '{} test'.format(name))
    plt.legend()
    plt.title('Power comparision with fixed test size 0.05')
    plt.savefig('./output/{}/{}_{}_power.pdf'.format(eval, dist_name, perturb))
    plt.close()

def experiments():
    # params = {
    #     'omega':np.array([0.2, 0.8]),
    #     'mean':np.array([0, 5]),
    #     'var':np.array([1, 1])
    # }
    # perturbs = ['mean', 'var', 'omega']
    xdim = 2
    hdim = 4
    # np.random.seed(666)
    params = {
        'W':np.random.randn(xdim, hdim),
        'b':np.random.randn(xdim),
        'c':np.random.randn(hdim)
    }
    RBM_experiments(params['W'], params['b'], params['c'])
    perturbs = ['W']
    for perturb in perturbs:
        simulation(m=1000, eval='perturbation', dist_name = 'gb_rbm', test_names=['HScore', 'Likelihood Ratio'], params = params, perturb=perturb)

def RBM_experiments(W, b, c):
    # xdim = 2
    # hdim = 4
    # W = np.random.randn(xdim, hdim)
    # b = np.random.randn(xdim)
    # c = np.random.randn(hdim)
    gb_rbm_model = GB_RBM(W, b, c)
    X = gb_rbm_model.sample(500, 100)
    sample_xaxis = [sample[0] for sample in X]
    sample_yaxis = [sample[1] for sample in X]
    xaxis, yaxis = np.mgrid[-5:5:.1, -5:5:.1]
    pos = np.dstack((xaxis, yaxis))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.contourf(xaxis, yaxis, np.exp(-gb_rbm_model.free_energy(pos)))
    ax2.scatter(sample_xaxis, sample_yaxis)
    plt.show()

if __name__ == '__main__':
    # RBM_experiments()
    experiments()