"""Distributions
1d mixture of gaussians
gaussian-bernoulli restricted bolzman machine

To do
---
multivariate gaussian
multivariate laplace
multivariate t
"""
import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt

class GaussianMixture(object):
    """1d mixture of gaussians"""
    def __init__(self, params):
        """Create a Gaussian Mixture Model"""
        self.weight = torch.exp(params['logweight'])
        self.mean = params['mean']
        self.logvar = params['logvar']
        self.model = [Normal(_mean, torch.sqrt(_var)) 
                        for (_mean, _var) in zip(self.mean, torch.exp(self.logvar))]
        self.params = [self.weight, self.mean, self.logvar]
    def train(self, param_train = 'mean'):
        if param_train == 'mean':
            self.mean.requires_grad_()
        if param_train == 'logvar':
            self.logvar.requires_grad_()
        if param_train == 'logweight':
            self.weight.requires_grad_()
        self.params = [self.weight, self.mean, self.logvar]
    def pdf(self, x, item = False):
        _probs = [w*torch.exp(m.log_prob(x)) 
                    for (w, m) in zip(self.weight, self.model)]
        if item:
            #for score and hyvarinen score calculation
            return _probs, sum(_probs)
        else:
            #sum(_probs) in shape Nx1
            return sum(_probs).sum(-1)
    def cdf(self, x):
        x = x.squeeze(-1)
        _cums = [w*m.cdf(x) 
                    for (w, m) in zip(self.weight, self.model)]
        return sum(_cums)
    def cdf_numpy(self, x):
        x = x.squeeze(-1)
        mcdf = 0.0
        for i in range(len(self.weight)):
            mcdf += self.weight[i].cpu().numpy() * stats.norm.cdf(x, loc=self.mean[i].cpu().numpy(), scale=np.sqrt(np.exp(self.logvar[i].cpu().numpy())))
        return mcdf
    def score(self, x):
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob*(-(x-mean) / var) 
                        for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        return mdpdf / mpdf
    def hscore(self, x):
        x = x.squeeze(-1)
        _probs, mpdf = self.pdf(x, item=True)
        mdpdf = sum([_prob*(-(x-mean) / var) 
                        for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        ddpdf = sum([_prob*((-(x-mean) / var)**2-1/var) 
                        for (_prob, mean, var) in zip(_probs, self.mean, torch.exp(self.logvar))])
        dlnpdf = mdpdf / mpdf
        hscore = -0.5*(dlnpdf**2)+ddpdf/mpdf
        return hscore
    def sample(self, num_samples):
        k = len(self.weight)
        samples = torch.empty(num_samples, k)
        for i in range(k):
            samples[:,i] = self.model[i].sample((num_samples,))
        mixture_idx = torch.multinomial(self.weight, num_samples=num_samples, replacement=True)
        samples = torch.gather(samples, -1, index=mixture_idx.unsqueeze(-1))
        return samples

class GaussBernRBM(object):
    """Gaussian-Bernoulli Restricted Boltzmann Machine
    
    Args:
        W (Tensor float64 in size (xdim, hdim)) parameter matrix for the visibles and the hiddens
        bx (Tensor float64 in size (xdim,)) biase for the visibles
        bh (Tensor float64 in size (hdim,)) biase for the hiddens
    """
    def __init__(self, params):
        """create a RBM"""
        self.W = params['W']
        self.bx = params['bx']
        self.bh = params['bh']
        self.xdim, self.hdim = self.W.shape
        self.params = [self.W, self.bx, self.bh]
    def train(self, samples, train_epoch, param_train_name, optimizer_name = 'SGD'):
        if param_train_name == 'W':
            self.W.requires_grad_()
            param_train = self.W
        elif param_train_name == 'bx':
            self.bx.requires_grad_()
            param_train = self.bx
        elif param_train_name == 'bh':
            self.bh.requires_grad_()
            param_train = self.bh
        else:
            raise ValueError('param_train_name')
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam([param_train], lr=0.001, betas=(0.9, 0.99))
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD([param_train], lr=0.001)
        else:
            raise ValueError('optimizer_name')
        
        #gradient descent
        for i in range(train_epoch):
            optimizer.zero_grad()
            hiddens = self.hidden_given_visible(samples)
            samples_gibbs = self.visible_given_hidden(hiddens)
            # loss = self.free_energy(samples).mean() - self.free_energy(samples_gibbs).mean()
            loss = self.hscore(samples_gibbs).mean()
            loss.backward()
            optimizer.step()
            # print('Epoch:{}'.format(i), 'Loss:{}'.format(loss))
        param_train.requires_grad_(False)
        # print('training Done')

    def free_energy(self, x):
        """Free energy function
            F(x) = -x`b+.5*(x`x+b`b)-\sum_hdim log(1+exp(self.bx+W`x))
        """
        x_term  = torch.matmul(x, self.bx)-0.5*torch.sum(x**2, dim=-1)-0.5*torch.sum(self.bx**2, dim=-1)
        w_x_h = F.linear(x, self.W.t(), self.bh)
        h_term = torch.sum(F.softplus(w_x_h), dim=-1)
        return -x_term - h_term
    def pdf(self, x):
        """Unnormalized probability
            \tilde(p)(x) = exp(-F(x))
        """
        return torch.exp(-self.free_energy(x))
    def score(self, x):
        """Score function for Gaussian-Bernoulli RBM,
            s(x) = b-x+sigmoid(xW+c)W`
        """
        sig = torch.sigmoid(F.linear(x, self.W.t(), self.bh))
        _x_px = F.linear(sig, self.W, self.bx-x)
        return _x_px
    def hscore(self, x):
        """Hyvarinen score for Gaussian-Bernoulli RBM,
            s_H(x) = 0.5*||grad_log_px||^2+trace(grad_grad_log_px)
        """
        sig = torch.sigmoid(F.linear(x, self.W.t(), self.bh))
        _x_px = F.linear(sig, self.W, self.bx-x)
        _x_sig = torch.matmul(sig.unsqueeze(-1), (1-sig).unsqueeze(1))
        _xx_px = -torch.sum(torch.matmul((1-sig)*(1+sig), self.W.t()**2), dim=-1)+self.bx.shape[0]
        # _xx_px = 1-torch.matmul(torch.matmul(self.W.unsqueeze(0), _x_sig), self.W.t().unsqueeze(0))
        # _xx_px = _xx_px.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1) #sum the trace
        hscore = 0.5*torch.sum(_x_px**2, dim=-1)-_xx_px
        return hscore
    def visible_given_hidden(self, h):
        """Conditional sampling visible variables given hidden variables"""
        mean_x_cond_h = F.linear(h, self.W, self.bx)
        sample_x = torch.randn(h.size()[0], self.xdim) + mean_x_cond_h
        return sample_x
    def hidden_given_visible(self, x):
        """Conditional sampling hidden variables given visible variables"""
        mean_h_cond_x = torch.sigmoid(F.linear(x, self.W.t(), self.bh))
        return mean_h_cond_x.bernoulli()
    def sample(self, num_samples, gibbs_iters=2000):
        """
        Gibbs sampling for Gaussian-Bernoulli RBM

        Args:
            num_samples (int): the sample size
            gibbs_iters (int): the iteration numbers in gibbs sampling
        
        Returns:
            Tensor: Visible samples
        """
        x = torch.randn(num_samples, self.xdim, dtype=torch.float64)
        h = torch.randint(1, (num_samples, self.hdim))
        for _ in range(gibbs_iters):
            h = self.hidden_given_visible(x)
            x = self.visible_given_hidden(h)
        return x

def visualize_1dGMM(save_path):
    params = {
        'logweight':torch.log(torch.tensor([0.2, 0.8])),
        'mean':torch.tensor([0., 5.]),
        'logvar':torch.tensor([0., 0.])
    }
    model0 = GaussianMixture(params)
    null_samples = model0.sample(1000)
    null_samples = null_samples.cpu().numpy()
    plt.hist(null_samples)
    plt.savefig(save_path+'/to_delete.png')

def visualize_RBM(save_path):
    """visualize Gaussian RBM samples
    """
    dx = 2
    dh = 10

    W = torch.randn(dx, dh, dtype=torch.float64)
    b = torch.randn(dx, dtype=torch.float64)
    c = torch.randn(dh, dtype=torch.float64)

    p_x = GaussBernRBM(W, b, c)
    p_samples = p_x.sample(1000)

    sample_xaxis = p_samples[:, 0]
    sample_yaxis = p_samples[:, 1] 
    xaxis, yaxis = np.mgrid[-10:10:.1, -10:10:.1]
    pos = np.dstack((xaxis, yaxis))
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    z = p_x.pdf(torch.from_numpy(pos).type(torch.float64))
    ax2.contourf(xaxis, yaxis, z)
    ax2.scatter(sample_xaxis, sample_yaxis, s=1)
    plt.savefig(save_path+'/to_delete.png')

####### To finish high-dimensional distributions
class MVN(object):
    def __init__(self, mean, cov):
        self.mean = mean
        self.bhov = cov
    def score(self, x):
        pass
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
    def sample(self, x):
        pass

class multivariate_Laplace(object):
    def __init__(self,loc,scale):
        self.comp=torch.distributions.laplace.Laplace(loc, scale)
        self.dim=loc.shape[0]
    def rsample(self,size):
        return self.comp.rsample(size)
    def log_prob(self,X):
        log_prob=torch.sum(self.comp.log_prob(X),dim=-1)
        return log_prob
    def score(self, x):
        pass
    def hscore(self, x):
        pass
    def sample(self,size):
        return self.rsample(torch.Size([size]))

class multivariate_t(object):
    def __init__(self,loc,df,scale=1):
        self.scale=scale
        self.loc=loc
        self.df=df
        # Create univariate student t distribution
        self.comp_dist=torch.distributions.studentT.StudentT(self.df,loc=self.loc,scale=self.scale)
    def rsample(self,size):
        # size should be N x D
        return self.comp_dist.rsample(size)
    def log_prob(self,X):
        log_prob=torch.sum(self.comp_dist.log_prob(X),dim=-1)
        return log_prob
    def score(self, x):
        pass
    def hscore(self, x):
        pass
    def sample(self,size):
        return self.rsample(torch.Size([size]))
