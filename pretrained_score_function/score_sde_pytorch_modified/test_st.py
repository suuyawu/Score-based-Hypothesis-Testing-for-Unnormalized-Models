"""
pretrain-trained neural network that estimates score_function \partial_x logp
"""
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
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


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

batch_size = 64 
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0 

sigmas = mutils.get_sigmas(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

#get the score_fn for input data
score_fn = mutils.get_score_fn(sde, state['model'], train=False, continuous=False) #score_fn(x, t=0)
def score_fn_0(x):
    return score_fn(x, t=torch.zeros(x.size()[0]))

#get input data
cifar10_data = datasets.CIFAR10(root='../data', 
                train = False, 
                download = True,
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                            ]))
eval_cf10 = torch.utils.data.DataLoader(cifar10_data, batch_size=batch_size, shuffle=True)

s_hscore_vrs_cf10 = []
for i, (batch_cf10, _) in enumerate(eval_cf10):
    batch_cf10 = batch_cf10.to(config.device)
    s_hscore_vr_cf10 = sliced_hscore_vr(score_fn_0, batch_cf10, n_particles=1)
    s_hscore_vrs_cf10.append(s_hscore_vr_cf10.item())
    if i == 100:
        print(f"CIFAR10 average sliced_score_vr: {s_hscore_vr_cf10.mean().item()}")
        # hscore_cf10 = hscore(score_fn_0, batch_cf10)
        # print(f"CIFAR10 hyverinen score: {hscore_cf10.mean().item()}")
        break

svhn_data = datasets.SVHN(root='../data', 
                split = 'test', 
                download = True,
                transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                # transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
                            ]))
s_hscore_vrs_svnh = []
eval_svnh = torch.utils.data.DataLoader(svhn_data, batch_size=64, shuffle=True)
for i, (batch_svnh, _) in enumerate(eval_svnh):
    batch_svnh = batch_svnh.to(config.device)
    s_hscore_vr_svnh = sliced_hscore_vr(score_fn_0, batch_svnh, n_particles=1)
    s_hscore_vrs_svnh.append(s_hscore_vr_svnh.item())
    if i == 100:
        print(f"SVNH average sliced_score_vr: {s_hscore_vr_svnh.mean().item()}")
        # hscore_svnh = hscore(score_fn_0, batch_svnh)
        # print(f"CIFAR10 hyverinen score: {torch.tensor(hscore_svnh).mean().item()}")
        break

plt.hist(np.array(s_hscore_vrs_cf10), label = 'ID-cf10')
plt.hist(np.array(s_hscore_vrs_svnh), label = 'OOD-svnh')
plt.legend()
plt.title('distribution of scores')
plt.savefig('../output/result1.pdf')