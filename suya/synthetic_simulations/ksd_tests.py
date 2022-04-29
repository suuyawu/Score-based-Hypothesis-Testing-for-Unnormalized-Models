"""
Reproduce implementation of MMD, KSD, and SKSD for goodness-of-fit tests
"""
import torch
import numpy as np
from ksd_utils import *

def KSD_test(x, score_function, width='median', nboot=1000):
    #set up the the bandwidth of RBF Kernel
    if width == 'median':
        h = np.sqrt(0.5 * find_median_distance(x))
    else:
        h = ratio_median_heuristic(x, score_function)
    #RBF kernel
    n_samples, xdim = x.shape
    _xscore = score_function(x)
    XX = np.dot(x, x.T)
    xs = np.expand_dims(np.sum(x*x, axis=1), axis= 1)
    x_xscore = np.sum(x*_xscore, axis=1)

    H = xs + xs.T - 2*XX
    Kxy = np.exp(-H/(2*h**2))
    sqxdy = -(np.dot(_xscore, x.T) - np.expand_dims(x_xscore, axis=1))/h**2
    dxsqy = sqxdy.T
    dxdy = (-H/h**4 + xdim/h**2)

    M = (np.dot(_xscore,_xscore.T) + sqxdy + dxsqy + dxdy) * Kxy
    M2 = M - np.diag(np.diag(M))
    ksd = np.sum(M2)/(n_samples*(n_samples-1))
    ksdV = np.sum(M) / (n_samples**2)
    #Bootstrap
    bootstrapSamples =  [np.NaN]*nboot
    for i in range(nboot):
        wts_boot = np.random.multinomial(1, [1/n_samples]*n_samples, n_samples)/n_samples
        bootstrapSamples[i] = np.dot(np.dot((wts_boot.T-1/n_samples),M2),(wts_boot-1/n_samples))
    p = np.mean(bootstrapSamples >= ksd)
    return p  

def KSD_test_power(XX, score_function, width='median', nboot=1000):
    pvalues = [KSD_test(X, score_function, width, nboot) for X in XX]
    rejects = [pvalue<0.05 for pvalue in pvalues]
    power = sum(rejects)/len(rejects)
    return power


if __name__ == '__main__':
    #test
    Z = np.arange(1, 91)
    Z.shape = (3,30)
    p = KSD_test(Z.T, score_function, width='median', nboot=1000)
    print(p)