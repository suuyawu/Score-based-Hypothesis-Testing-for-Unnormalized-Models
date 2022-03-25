"""
pretrain-trained neural network that estimates score_function \partial_x logp
"""
from ast import Pass
from multiprocessing.sharedctypes import Value
import seaborn as sns
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from sde_lib import VESDE, VPSDE

from utils_st import *
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from likelihood import get_likelihood_fn

import os
import math
from torchvision.utils import save_image
from torch.utils.data import Subset

# def preprocessing(data, sigma, iters):

#   for i in iters:
#     noise_std = torch.Tensor([2*group['lr']])
#     noise_std = noise_std.sqrt()
#     noise = p.data.new(
#     p.data.size()).normal_(mean=0, std=1)*noise_std
#     p.data.add_(noise)

def get_subindices(dataset, data_size):
    import random
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return indices[:data_size]

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

def hist_plot(arrs, labels, save_path, title):
  """
  histgram of distribution
  input:
    arrs = [in_score, ood_score]
    labels = [label1, label2]
  """
  for (arr, label) in zip(arrs, labels):
    plt.hist(arr, label = label, alpha=0.8)
  plt.xlabel("{}".format(label))
  plt.ylabel("frequency")
  plt.legend(loc="upper right")
  plt.title(title)
  plt.savefig(save_path+'.png')
  plt.close()

def roc_plot(scores, labels, save_path, title):
  """
  roc curve
  input:
    scores=[[id_score1, ood_score1], [id_score2, ood_score2], ...]
    labels = [label1, label2, ...]
  """
  from sklearn.metrics import roc_curve, auc
  
  for i, score in enumerate(scores):
    score_id, score_ood = score[0], score[1]
    y_true = np.array([0]*len(score_id)+[1]*len(score_ood))
    score_arr = np.append(score_id, score_ood)
    fpr, tpr, _ = roc_curve(y_true, score_arr, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=labels[i]+"(area = %0.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], linestyle="--")
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title(title)
  plt.legend(loc="lower right")
  plt.savefig(save_path+'_roc.png')
  plt.close()

def get_hscore_batch(dataset, data_name, batch_size, batch_mean=False, save=True, load=True):
  """
  calculate hyvarian score(bits/dim) for data mini-batch
  input:
    batch_mean: output the batch means
  """
  save_path = '../output/vd/'+data_name+'_hscore_fd.npy'
  if os.path.isfile(save_path) and load:
    arr = np.load(save_path)
    if len(arr) >= len(dataset):
      return arr

  data_load = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
  data_dim = np.prod(next(iter(data_load))[0].shape[2:])
  hscore_fds = np.array([])
  for i, (data_batch, _) in enumerate(data_load):
    data_batch = data_batch.to(config.device)
    hscore_fd_ = hscore_fd(score_fn_0, data_batch)
    if batch_mean:
      hscore_fds = np.append(hscore_fds, hscore_fd_.mean().cpu().numpy() / (data_dim*np.log(2)))
    else:
      hscore_fds = np.append(hscore_fds, hscore_fd_.cpu().numpy() / (data_dim*np.log(2)))
  if save:
    np.save(save_path, hscore_fds)
  return hscore_fds

def get_nll_batch(dataset, data_name, batch_size, batch_mean=False, save=True, load=True):
  """
  calculate negative log likelihood(bits/dim) for data mini-batch
  input:
    batch_mean: output the batch means
  """
  save_path = '../output/vd/'+data_name+'_nll.npy'
  if os.path.isfile(save_path) and load:
    arr = np.load(save_path)
    if len(arr) >= len(dataset):
      return arr

  data_load = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
  nlls = np.array([])
  for i, (data_batch, _) in enumerate(data_load):
    data_batch = data_batch.to(config.device)
    nll_, _, _ = likelihood_fn(score_model, data_batch)
    if batch_mean:
      nlls = np.append(nlls, nll_.mean().cpu().numpy())
    else:
      nlls = np.append(nlls, nll_.cpu().numpy())
  if save:
    np.save(save_path, nlls)
  return nlls

def test_st_lrt_batch(id_dataset, ood_dataset, ood_name, batch_size, lrt=True, batch_mean=False, save=True):
  """
  Compare lrt and st for ID:cifar10 v.s. OOD:svhn/fakecifar10 (DATA SUBSET)
  """
  path = os.path.join('../output/vd', ood_name)
  id_loader = torch.utils.data.DataLoader(id_dataset, batch_size, shuffle=True)
  ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size, shuffle=True)

  hscore_fds_id = np.array([])
  hscore_fds_ood = np.array([])    
  if lrt:
    likelihood_id = np.array([])
    likelihood_ood = np.array([])
  id_dim = np.prod(next(iter(id_loader))[0].shape[2:])
  ood_dim = np.prod(next(iter(ood_loader))[0].shape[2:])

  for i, zipdata in enumerate(tqdm(zip(id_loader, ood_loader))):
    id_batch = zipdata[0][0].to(config.device)
    ood_batch = zipdata[1][0].to(config.device)
    
    hscore_fd_id = hscore_fd(score_fn_0, id_batch)
    hscore_fd_ood = hscore_fd(score_fn_0, ood_batch)
    if batch_mean:
      hscore_fds_id = np.append(hscore_fds_id, hscore_fd_id.mean().cpu().numpy() / (id_dim*np.log(2)))
      hscore_fds_ood = np.append(hscore_fds_ood, hscore_fd_ood.mean().cpu().numpy() / (ood_dim*np.log(2)))
    else:
      hscore_fds_id = np.append(hscore_fds_id, hscore_fd_id.cpu().numpy() / (id_dim*np.log(2)))
      hscore_fds_ood = np.append(hscore_fds_ood, hscore_fd_ood.cpu().numpy() / (ood_dim*np.log(2)))

    label1 = 'ID-cf10-st_fd'
    label2 = 'OOD-{}-st_fd'.format(ood_name)
    save_path = path+ '_st_dist'
    if batch_mean:
      save_path = save_path+ '_mean'
    title = 'Distribution of HScore(bits/dim) for cifar10 v.s. {name} with {num} samples'.format(name = ood_name, num = i*batch_size+len(id_batch))
    hist_plot([hscore_fds_id, hscore_fds_ood], [label1, label2], save_path = save_path, title = title)

    if lrt:
      bpd_id, _, _ = likelihood_fn(score_model, id_batch)
      bpd_ood, _, _ = likelihood_fn(score_model, ood_batch)
      if batch_mean:
        likelihood_id = np.append(likelihood_id, bpd_id.mean().cpu().numpy())
        likelihood_ood = np.append(likelihood_ood, bpd_ood.mean().cpu().numpy())
      else:
        likelihood_id = np.append(likelihood_id, bpd_id.cpu().numpy())
        likelihood_ood = np.append(likelihood_ood, bpd_ood.cpu().numpy())

      label1 = 'ID-cf10-lrt'
      label2 = 'OOD-{}-lrt'.format(ood_name)
      save_path = path+ '_lrt_dist'
      if batch_mean:
        save_path = save_path + '_mean'
      title = 'Distribution of NLL(bits/dim) for cifar10 v.s. {name} with {num} samples'.format(name = ood_name, num = i*batch_size+len(id_batch))
      hist_plot([likelihood_id, likelihood_ood], [label1, label2], save_path = save_path, title = title)

    #ROC curve
    scores = [[hscore_fds_id, hscore_fds_ood],]
    labels = ['st',]
    roc_plot(scores, labels, path, "ROC Curve of Test Performance")
    if lrt:
      scores.append([likelihood_id, likelihood_ood])
      labels.append('lrt')
      roc_plot(scores, labels, path, "ROC Curve of Test Performance")
  
  if save:
    sp_id = '../output/vd/'+'cifar10_hscore_fd.npy'
    if not os.path.isfile(sp_id):
      np.save(sp_id, hscore_fds_id)
    sp_ood = '../output/vd/'+ood_name+'_hscore_fd.npy'
    if not os.path.isfile(sp_ood):
      np.save(sp_ood, hscore_fds_ood)
    
    if lrt:
      sp_id = '../output/vd/'+'cifar10_nll.npy'
      if not os.path.isfile(sp_id):
        np.save(sp_id, likelihood_id)
      sp_ood = '../output/vd/'+ood_name+'_nll.npy'
      if not os.path.isfile(sp_ood):
        np.save(sp_ood, likelihood_ood)


def test_ood_cifar10(ood_name, data_size, batch_size, lrt=True, batch_mean=False, bybatch=True, crop=False, resize = False):
  """
  Visualize HScore statistics for ID:cifar10 v.s. OOD:svhn, lcun, tiny-imagenet, ... (SUBSET of FULL TEST DATASET)
  """
  path = '../output/vd/'+ood_name
  #load data
  transform_ = transforms.Compose([transforms.ToTensor(),])
  if crop:
    transform_ = transforms.Compose([transforms.ToTensor(),transforms.CentreCrop(32),])
  if resize:
    transform_ = transforms.Compose([transforms.ToTensor(),transforms.Resize(32),])
  cifar10_data = datasets.CIFAR10(root='../data/CIFAR10/test', train = False, download = True, transform = transforms.Compose([transforms.ToTensor(),]))
  if ood_name == 'svhn':
    ood_data = datasets.SVHN(root='../data/SVHN/test', split = 'test', download = True, transform = transform_)
  elif ood_name == 'tiny-imagenet':
    ood_data = datasets.ImageFolder('../data/Tiny-ImageNet/tiny-imagenet-200/test', transform = transform_)
  elif ood_name == 'cifar100':
    ood_data = datasets.CIFAR100(root='../data/CIFAR100/test', train=False, download=True, transform=transform_)
  elif ood_name == 'lsun':
    pass
  elif ood_name == 'mnist':
    pass
    ood_data = datasets.MNIST(root='../data/MNIST/test', train=False, download=True, transform=transform_)
  elif ood_name == 'fashion-mnist':
    pass
    ood_data = datasets.MNIST(root='../data/MNIST/test', train=False, download=True, transform=transform_)
  elif ood_name == 'omniglot':
    pass
    ood_data = datasets.Omniglot(root='../data/Omniglot/test', background=True, download=False, transform=transform_)
  else:
    raise ValueError('OOD DATA TYPE IS NOT SUPPORT!')

  cifar10_subdata = Subset(cifar10_data, get_subindices(cifar10_data, data_size))
  ood_subdata = Subset(ood_data, get_subindices(ood_data, data_size))
  
  if bybatch:
    test_st_lrt_batch(cifar10_subdata, ood_subdata, ood_name, batch_size, lrt, batch_mean)
  else:
    cifar10_hscore = get_hscore_batch(cifar10_subdata, 'cifar10', batch_size, batch_mean, save=True, load=True)
    ood_hscore = get_hscore_batch(ood_subdata, ood_name, batch_size, batch_mean, save=True, load=True)
    title = 'Distribution of HScore(bits/dim) for cifar10 v.s. {name} with {num} samples'.format(name=ood_name, num=data_size)
    save_path = path+'_st_dist'
    if batch_mean:
      save_path = save_path + '_mean'
    hist_plot([cifar10_hscore, ood_hscore], ['cifar10', ood_name], save_path, title)
    scores = [[cifar10_hscore, ood_hscore],]
    labels = ['st',]
    roc_plot(scores, labels, '../output/vd/'+ood_name, "ROC Curve of Test Performance")

    if lrt:
      cifar10_nll = get_nll_batch(cifar10_subdata, 'cifar10', batch_size, batch_mean, save=True, load=True)
      ood_nll = get_nll_batch(ood_subdata, ood_name, batch_size, batch_mean, save=True, load=True)
      title = 'Distribution of NLL(bits/dim) for cifar10 v.s. {name} with {num} samples'.format(name=ood_name, num=data_size)
      save_path = path+'_lrt_dist'
      if batch_mean:
        save_path = save_path + '_mean'
      hist_plot([cifar10_nll, ood_nll], ['cifar10', ood_name], save_path, title)
      scores.append([cifar10_nll, ood_nll])
      labels.append('lrt')
      roc_plot(scores, labels, path, "ROC Curve of Test Performance")

def test_fake_cifar10(data_size, batch_size, names = ['WGAN-GP', 'ReACGAN', 'StyleGAN2-ADA', 'NCSN++'], lrt=True, batch_mean=False):
  path='../output/CIFAR10_fake/'
  #true data score
  true_data = datasets.CIFAR10(root='../data/CIFAR10/test/', train = False, download = True, transform = transforms.Compose([transforms.ToTensor(),]))
  true_subdata = Subset(true_data, get_subindices(true_data, data_size))
  hscore_fd_true = get_hscore_batch(true_subdata, 'cifar10', batch_size, batch_mean, save=True, load=True)
  if lrt:
    nll_true = get_nll_batch(true_subdata, 'cifar10', batch_size, batch_mean, save=True, load=True)
  
  scores = []
  labels = []

  for name in names:
    #load fake data
    data_path = path+name+'/'
    if name == 'NCSN++':
      data_path = data_path = path+name+'/nscaled/'
    if name == 'WGAN-GP':
      fid = 25.852
    elif name == 'ReACGAN':
      fid = 7.792
    elif name == 'StyleGAN2-ADA':
      fid = 2.42
    elif name == 'NCSN++':
      fid = 2.38
    else:
      raise ValueError("FAKE IMAGE TYPE IS NOT SUPPORTED")
    #fake data score
    fake_data = datasets.ImageFolder(data_path, transform = transforms.Compose([transforms.ToTensor(),]))
    fake_subdata = Subset(fake_data, get_subindices(fake_data, data_size))
    hscore_fd_fake = get_hscore_batch(fake_subdata, name, batch_size, batch_mean, save=True, load=True)
    hist_plot([hscore_fd_true, hscore_fd_fake], ['cifar10', name+'_FID={}'.format(fid)], path+name+'_st_dist', "Distribution of HScore(bits/dim) with {} samples".format(data_size))
    scores.append([hscore_fd_true, hscore_fd_fake])
    labels.append('st'+name+'_FID={}'.format(fid))
    roc_plot(scores, labels, path, "ROC Curve of Test Performance")

    if lrt:
      nll_fake = get_nll_batch(fake_subdata, name, batch_size, batch_mean, save=True, load=True)
      hist_plot([nll_true, nll_fake], ['cifar10', name+'_FID={}'.format(fid)], path+name+'_lrt_dist', "Distribution of NLL(bits/dim) with {} samples".format(data_size))
      scores.append([nll_true, nll_fake])
      labels.append('lrt'+name+'_FID={}'.format(fid))
      roc_plot(scores, labels, path, "ROC Curve of Test Performance")

def sample_fake(num_images, batch_size, path):
  import sampling
  from sampling import (ReverseDiffusionPredictor, 
                        LangevinCorrector, 
                        EulerMaruyamaPredictor, 
                        AncestralSamplingPredictor, 
                        NoneCorrector, 
                        NonePredictor,
                        AnnealedLangevinDynamics)

  img_size = config.data.image_size
  channels = config.data.num_channels
  shape = (batch_size, channels, img_size, img_size)
  predictor = ReverseDiffusionPredictor
  corrector = LangevinCorrector
  snr = 0.16
  n_steps = 1
  probability_flow = False
  num_batches = int(num_images/batch_size)+1

  sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                        inverse_scaler, snr, n_steps=n_steps,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        eps=sampling_eps, device=config.device)

  with torch.no_grad():
    for i in tqdm(range(0, num_batches), disable=False):
      fake_cifar10_batch, _ = sampling_fn(score_model)
      for idx, img in enumerate(fake_cifar10_batch.detach()):
        if batch_size * i + idx < num_images:
            # save_image(((img+1)/2).clamp(0.0, 1.0), os.path.join(path, "{idx}.png".format(idx=batch_size * i + idx)))
            save_image(img, os.path.join(path, "{idx}.png".format(idx=batch_size * i + idx)))

if __name__ == "__main__":
  sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
  if sde.lower() == 'vesde':
    from configs.ve import cifar10_ncsnpp_continuous as configs
    ckpt_filename = "../output/exp/vesde/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    config = configs.get_config()  
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif sde.lower() == 'vpsde':
    from configs.vp import cifar10_ddpmpp_continuous as configs  
    ckpt_filename = "../output/exp/vpsde/cifar10_ddpmpp_continuous/checkpoint_8.pth"
    config = configs.get_config()
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3

  config.training.batch_size = 64
  config.eval.batch_size = 64
  scaler = get_data_scaler(config)
  inverse_scaler = get_data_inverse_scaler(config)
  score_model = mutils.create_model(config)

  optimizer = get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(),decay=config.model.ema_rate)
  state = dict(step=0, optimizer=optimizer,model=score_model,ema=ema)
  state = restore_checkpoint(ckpt_filename,state,config.device)
  ema.copy_to(score_model.parameters())
  
  score_fn = mutils.get_score_fn(sde, state['model'], train=False, continuous=False) #score_fn(x, t=0)
  def score_fn_0(x):
      return score_fn(x, t=torch.zeros(x.size()[0]))
  likelihood_fn = get_likelihood_fn(sde, inverse_scaler, eps=1e-5)

  # sample_fake(num_images=10000, batch_size = 1024, path = '../output/CIFAR10_fake/NCSN++/fake/')
  # test_ood_cifar10(ood_name = 'svhn', data_size = 10000, batch_size = 128, lrt=True, batch_mean=False)
  # test_ood_cifar10(ood_name='tiny-imagenet', data_size=10000, batch_size=128, lrt=True, batch_mean=False, bybatch=True, resize=True)
  # test_ood_cifar10(ood_name='cifar100', data_size=10000, batch_size=128, lrt=True, batch_mean=False, bybatch=True)
  test_fake_cifar10(data_size=10000, batch_size=128, names = ['WGAN-GP', 'ReACGAN', 'StyleGAN2-ADA', 'NCSN++'], lrt=True, batch_mean=False)
