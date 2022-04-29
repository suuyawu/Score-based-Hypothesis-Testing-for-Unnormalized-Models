"""
one-dimensional mixture of gaussians, exponential distribution, chi, chi2, weibull
multivariate normal distribution
"""
from pickle import FALSE
from importlib_metadata import MetadataPathFinder
import numpy as np
from numpy import matlib as mb
import torch
import scipy

class OneDimensionGM():
    def __init__(self, omega, mean, var):
        self.omega = omega
        self.mean = mean
        self.var = var
    def pdf(self, x, density=False):
        mpdf = 0.0
        mdpdf = 0.0
        ddpdf = 0.0
        for i in range(len(self.omega)):
            pdf_ = scipy.stats.norm.pdf(x, loc=self.mean[i], scale=np.sqrt(self.var[i]))
            mpdf += self.omega[i] * pdf_
            mdpdf += self.omega[i] * pdf_ * (-(x-self.mean[i]) / self.var[i])
            ddpdf += self.omega[i] * pdf_ * ((-(x-self.mean[i]) / self.var[i])**2-1/self.var[i])
        if density:
            return mpdf
        return mpdf, mdpdf, ddpdf
    def cdf(self, x):
        mcdf = 0.0
        for i in range(len(self.omega)):
            mcdf += self.omega[i] * scipy.stats.norm.cdf(x, loc=self.mean[i], scale=np.sqrt(self.var[i]))
        return mcdf
    def hscore(self, x):
        mpdf, mdpdf, ddpdf = self.pdf(x)
        dlnpdf = mdpdf / mpdf
        hscore = -(mdpdf/mpdf)**2+ddpdf/mpdf+0.5*(dlnpdf**2)
        return hscore
    def sample(self, m, n):
        XX = np.zeros((m,n))
        theta = [[self.mean[t], np.sqrt(self.var[t])] for t in range(len(self.omega))]
        for i in range(m):
            mixture_idx = np.random.choice(len(self.omega), size=n, replace=True, p=self.omega)
            XX[i] = np.fromiter((np.random.normal(*(theta[j])) for j in mixture_idx), dtype=np.float64)
        return XX

class GB_RBM():
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine
    x: the observations, real-valued in R^xdim
    h: the hidden states, binary-valued in R^hdim
    """
    def __init__(self, W, b, c):
        """
        Gaussian Bernoulli RBM model

        Parameters:
            W: ndarray in shape (xdim, hdim)
            b: 1darray in shape (1, xdim)
            c: 1darray in shape (1, hdim)
        """
        self.xdim, self.hdim = W.shape
        self.W = W
        self.b = b
        self.c = c
    def _sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def free_energy(self, x):
        """
        Free energy F(x), defined as,
            F(x) = -log sum_h e^{-E(x, h)}
        For Gaussian-Bernoulli RBM, 
            F(x) = -x^Tb+1/2(x^Tx+b^Tb)-sum_hdim log(1+exp(self.b+W^Tx))
        """
        return -np.dot(x, self.b) +0.5*((x*x).sum(-1)+np.dot(self.b, self.b.T))- np.log(1+np.exp(np.dot(x, self.W)+self.c)).sum(-1)
    def hscore(self, x):
        """
        Hyvarinen score for Gaussian-Bernoulli RBM,
            s_H(x) = 
        """
        sig = self._sigmoid(np.dot(x, self.W)+self.c)
        _x_px = self.b-x+np.dot(sig, self.W.T)
        _xx_px = np.trace(1-np.dot(np.dot(self.W, np.dot(sig.T, 1-sig)), self.W.T))
        return 0.5*np.sum(_x_px*_x_px, axis=-1)-_xx_px
    def score_function(self, x):
        """
        score function for Gaussian-Bernoulli RBM,
            s(x) = 
        """
        sig = self._sigmoid(np.dot(x, self.W)+self.c)
        _x_px = self.b-x+np.dot(sig, self.W.T)
        return _x_px
    def pdf(self, x, density=True):
        return np.exp(-self.free_energy(x))
    def reject_sampling(self, num_samples, scaler, stop = 1e3):
        """
        Draw exact samples from Gaussian-Bernoulli RBM
        """
        samples = []
        stop_sign = 0
        from scipy.stats import multivariate_normal
        while len(samples)<num_samples and stop_sign<stop:
            mean = np.zeros(self.xdim)
            var = np.eye(self.xdim)
            x = np.random.multivariate_normal(mean, var)
            envelope = scaler * multivariate_normal.pdf(x, mean, var)
            p = np.random.uniform(0, envelope)
            if p < np.exp(-self.free_energy(x)):
                samples.append(x)
        return samples
    def visible_given_hidden(self, h):
        """
        Input the hiddens: ndarray in (n_samples, hdim)
        Output the visibles: ndarray in (n_samples, xdim)
        """
        m, n, _ = h.shape
        mean_v_h = np.dot(h, self.W.T) + self.b
        return np.random.randn(m, n, self.xdim)+mean_v_h
    def hidden_given_visible(self, x):
        """
        Input the visibles: ndarray in (n_samples, xdim)
        Output the hiddens: ndarray in (n_samples, hdim)
        """
        m, n, _ = x.shape
        mean_h_v = self._sigmoid(np.dot(x, self.W) + self.c)
        return  mean_h_v > np.random.rand(m, n, self.hdim)
    def sample(self, m, n, iters=50):
        """
        Draw samples from Gaussian-Bernoulli RBM by gibbs sampling
        """
        x=torch.randn(m, n, self.xdim)
        h=np.random.randint(2,(m, n, self.hdim))
        for t in range(iters):
            h = self.hidden_given_visible(x)
            x = self.visible_given_hidden(h)
        return x

class OneDimensionNormal():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def dlnprob(self, x):
        return None 
    def MGprob(self, x):
        from scipy.stats import norm
        return norm.pdf(x, self.mean, self.std)
    def lnprob(self, x):
        return  -(x-self.mean) **2 / 2*(self.std **2) - np.log(self.std**2)/2
    def hscore(self, x):
        return (x-self.mean) **2 / 2*(self.std **4) - 1/(self.std **2)
    def sample(self, m, n):
        return np.random.normal(self.mean, self.std, (m, n))

class MVN():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    def dlnprob(self, x):
        return -1 * np.matmul((x - self.mean), np.linalg.inv(self.cov))
    def MGprob(self, x):
        from scipy.stats import multivariate_normal
        mvn = multivariate_normal(mean=self.mean, cov=self.cov)
        return mvn.pdf(x)
    def lnprob(self, x):
        # lnprob_v = -0.5*(np.log(np.linalg.det(self.cov))  + (x - self.mean).transpose() * np.linalg.inv(self.cov) * (x - self.mean) + 2 * np.log(2 * np.pi))
        return np.mean(np.log(self.MGprob(x)))
    def hscore(self, x):
        invcov = np.linalg.inv(self.cov)
        t1 = 0.5* np.matmul(np.matmul(np.matmul((x - self.mean), invcov),invcov),(x - self.mean).transpose())
        t2 = - invcov.diagonal().sum() 
        t1 = t1.diagonal().sum()
        return t1/x.shape[0] + t2

class OneDimensionEXPON():
    def __init__(self, scale):
        self.scale = scale
    def dlnprob(self, x):
        return -np.ones(x.shape) * self.scale
    def MGprob(self, x):
        from scipy.stats import expon
        expn_ = expon()
        return expn_.pdf(x) / self.scale
    def lnprob(self, x):
        from scipy.stats import expon
        expn_ = expon()
        return np.mean(expn_.logpdf(x)*self.scale+np.log(self.scale))
    def hscore(self, x):
        return 0.5*(self.scale ** 2)

class OneDimensionCHI():
    def __init__(self, df, loc, scale):
        self.df = df # df = 1 is halfnorm; df = 2 is rayeigh; df = 3 is maxwell
        self.scale = scale
        self.loc = loc
    def dlnprob(self, x):
        return (self.df-1.0) / x - x
    def MGprob(self, x):
        from scipy.stats import chi
        chi_ = chi(df = self.df, loc = self.loc, scale = self.scale)
        return chi_.pdf(x)
    def lnprob(self, x):
        from scipy.stats import chi
        chi_ = chi(df = self.df, loc = self.loc, scale = self.scale)
        return np.mean(chi_.logpdf(x))
    def hscore(self, x):
        score = 0.5*((self.df-1.0) / x - x) **2 - (self.df-1.0) / x**2 -1
        return np.mean(score)

class OneDimensionCHI2():
    def __init__(self, df, loc, scale):
        self.df = df 
        self.scale = scale
        self.loc = loc
    def dlnprob(self, x):
        return (self.df/2-1.0) / x - 0.5
    def MGprob(self, x):
        from scipy.stats import chi2
        chi2_ = chi2(df = self.df, loc = self.loc, scale = self.scale)
        return chi2_.pdf(x)
    def lnprob(self, x):
        from scipy.stats import chi2
        chi2_ = chi2(df = self.df, loc = self.loc, scale = self.scale)
        return np.mean(chi2_.logpdf(x))
    def hscore(self, x):
        score = 0.5*((self.df/2-1.0) / x - 0.5) **2 - (self.df-1.0) / x**2
        score[np.isinf(score)] = 1e10
        score[np.isnan(score)] = 1e-10
        return np.mean(score)

class WeibullMin():
    def __init__(self, c, loc, scale):
        self.c = c
        self.loc = loc
        self.scale = scale
    def dlnprob(self, x):
        dlnprob_ = (self.c-1) /x -self.c*np.power(x, self.c-1)
        dlnprob_[np.isinf(dlnprob_)] = 1e10
        dlnprob_[np.isnan(dlnprob_)] = 1e-10
        return dlnprob_
    def MGprob(self, x):
        from scipy.stats import weibull_min
        wmin_ = weibull_min(c = self.c, loc = self.loc, scale = self.scale)
        return wmin_.pdf(x)
    def lnprob(self, x):
        from scipy.stats import weibull_min
        wmin_ = weibull_min(c = self.c, loc = self.loc, scale = self.scale)
        return np.mean(wmin_.logpdf(x))
    def hscore(self, x):
        dlnprob_ = self.dlnprob(x)
        score = 0.5*dlnprob_ **2 - (self.c-1)*(1/x**2 -self.c*np.power(x, self.c-2))
        score[np.isinf(score)] = 1e10
        score[np.isnan(score)] = 1e-10
        return np.mean(score)

class OneDimensionNk():
    def __init__(self, c):
        self.c = c
    def dlnprob(self, x):
        dlnprob_ = np.zeros(x.shape)
        for i in range(len(self.c)):
            dlnprob_ += self.c[i]*np.power(x,i-1)
        return dlnprob_
    def MGprob(self, x):
        return None
    def lnprob(self, x):
        return None
    def hscore(self, x):
        score = np.zeros(x.shape)
        dlnprob_ = self.dlnprob(x)
        for i in range(1, len(self.c)):
            score += self.c[i]*np.power(x,i-2)
        score = 0.5*dlnprob_**2 + score
        return np.mean(score)