from joblib import PrintTime
import torch
import torch.autograd as autograd
from tqdm import tqdm

def hscore(score_net, x):
    """
    calculate hscore exactly
    input:
        score_net: \partial_x logp(x) in shape (n, d)
        x: sample mini-batches in shape (n, d)
    output:
        hscore in shape (n, d)
    """
    x.requires_grad_(True)
    xlogp_ = score_net(x)
    loss1 = (torch.sum(xlogp_ * xlogp_, dim=tuple(range(1, len(x.shape)))) / 2.).detach()
    loss2 = torch.zeros(x.shape[0])

    for i in tqdm(range(x.shape[1])): #this for loop need to be modified for 4-dimensional data
        xxlogp_ = autograd.grad(xlogp_[:,i].sum(), x)[0][:,i]
        loss2 += xxlogp_.detach()

    x.requires_grad_(False)
    hscore = loss1 + loss2
    return hscore.mean()

def hscore_hutchinson(score_net, x, n_particles=1):
    """
    calculate hscore by Hutchinson's trick (Gaussian randoms)
    input:
        score_net: \partial_x logp(x) in shape (n, d1, d2, d3), n_particles << d1*d2*d3
        x: sample mini-batches in shape (n, d1, d2, d3)
        n_particles: number of random projections
    output:
        hscore in shape (n, d1, d2, d3)
    """
    #The variance using the Rademacher distribution (Hutchinson's estimator) 
    #has lower variance compared to the Gaussian, and is one reason it is often preferred.
    #But the number of Gaussian samples needed to obtain an error is smaller than Rademacher samples
    with torch.enable_grad():
        dup_x = x.unsqueeze(0).expand(n_particles, *x.shape).contiguous().view(-1, *x.shape[1:])
        dup_x.requires_grad_(True)

        xlogp_ = score_net(dup_x)
        vectors = torch.randn_like(dup_x, device=xlogp_.device)
        gradv = torch.sum(xlogp_ * vectors)
        grad2 = autograd.grad(gradv, dup_x)[0].to(xlogp_.device)
    x.requires_grad_(False)
    loss1 = (torch.sum(xlogp_ * xlogp_, dim=tuple(range(1, len(x.shape)))) / 2.).detach()
    loss2 = (torch.sum(vectors * grad2, dim=tuple(range(1, len(x.shape))))).detach()
    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    hscore = loss1 + loss2
    return hscore.mean()

def hscore_fd(score_net, x, eps=0.1):
    """
    calculate hscore by finite difference (Gaussian randoms)
    input:
        score_net: \partial_x logp(x) in shape (n, d1, d2, d3), n_particles << d1*d2*d3
        x: sample mini-batches in shape (n, d1, d2, d3)
        eps: magnitude for perturbation
    output:
        hscore in shape (n, d1, d2, d3)
    """
    dim = x.reshape(x.shape[0], -1).shape[-1]
    xlogp_ = score_net(x)
    vectors = torch.randn_like(x)
    #Scale the variance of vector to be (eps**2*I_{dxd})/dim
    vectors = vectors / torch.sqrt(torch.sum(vectors ** 2, dim=tuple(range(1, len(x.shape))), keepdim=True)) * eps
    out1 = score_net(x + vectors)
    out2 = score_net(x - vectors)
    grad2 = out1 - out2
    
    # loss_1 = torch.sum((grad1 * grad1) / 8.0, dim=tuple(range(1, len(x.shape))))
    loss_1 = (torch.sum(xlogp_ * xlogp_, dim=tuple(range(1, len(x.shape)))) / 2.).detach()
    loss_2 = (torch.sum(grad2 * vectors * (dim / (2  * eps *eps)), dim=tuple(range(1, len(x.shape))))).detach()
    hscore = (loss_1 + loss_2).mean()
    return hscore

def test():
    d = 1000
    mean = torch.rand((1, d))

    def score_net_toy(x, mean = mean):
        xlogp_ = -(x-mean)
        return xlogp_
    
    x = torch.randn(10000, d) + mean
    exact_hscore_ = -x.size()[1]+(torch.norm(x-mean, dim=-1)**2 /2).mean()

    hscore_ = hscore(score_net_toy, x)
    sliced_hscore_vr_ = hscore_hutchinson(score_net_toy, x, n_particles=1)
    esm_hscore_vr_ = hscore_fd(score_net_toy, x)
    print(hscore_, sliced_hscore_vr_, esm_hscore_vr_)
    print("exact h_score", torch.dist(hscore_, exact_hscore_))
    print("sliced h_score vr estimation", torch.dist(sliced_hscore_vr_, exact_hscore_))
    print("finite difference hscore", torch.dist(esm_hscore_vr_, exact_hscore_))