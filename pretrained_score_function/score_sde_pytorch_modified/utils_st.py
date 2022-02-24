import torch
import torch.autograd as autograd
from tqdm import tqdm

def hscore(score_net, x):
    """
    score_net: \partial_x logp(x) in shape (n, d)
    x: sample mini-batches in shape (n, d)
    """
    x.requires_grad_(True)
    xlogp_ = score_net(x)
    loss1 = (torch.sum(xlogp_ * xlogp_, dim=tuple(range(1, len(x.shape)))) / 2.).detach()
    loss2 = torch.zeros(x.shape[0])

    for i in tqdm(range(x.shape[1])): #this for loop need to be modified for 4-dimensional data
        xxlogp_ = autograd.grad(xlogp_[:,i].sum(), x)[0][:,i]
        loss2 += xxlogp_.detach()

    x.requires_grad_(False)
    loss = loss1 + loss2
    return loss.mean()

def sliced_hscore_vr(score_net, x, n_particles=1):
    """
    score_net: \partial_x logp(x) in shape (n, d1, d2, d3), n_particles << d2*d3
    x: sample mini-batches in shape (n, d1, d2, d3)
    n_particles: number of random projections
    """
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

    loss = loss1 + loss2
    return loss.mean()

def test():
    d = 1000
    mean = torch.rand((1, d))

    def score_net_toy(x, mean = mean):
        xlogp_ = -(x-mean)
        return xlogp_
    
    x = torch.randn(10000, d) + mean
    exact_hscore_ = -x.size()[1]+(torch.norm(x-mean, dim=-1)**2 /2).mean()

    hscore_ = hscore(score_net_toy, x)
    sliced_hscore_vr_ = sliced_hscore_vr(score_net_toy, x, n_particles=10)
    print("exact h_score", torch.dist(hscore_, exact_hscore_))
    print("sliced h_score vr estimation", torch.dist(sliced_hscore_vr_, exact_hscore_))