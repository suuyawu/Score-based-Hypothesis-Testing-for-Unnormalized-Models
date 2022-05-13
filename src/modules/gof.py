import copy
import datetime
import time
import sys
import torch
import models
import numpy as np
from config import cfg
from data import make_data_loader
from utils import to_device, collate, make_optimizer, make_scheduler
from .nonparam import CVM, KS
from .ksd import KSD
from .mmd import MMD
from .lrt import LRT
from .hst import HST


class GoodnessOfFit:
    def __init__(self, test_mode, alter_num_samples, alter_noise, alpha=0.05):
        self.test_mode = test_mode
        self.alter_num_samples = alter_num_samples
        self.alter_noise = alter_noise
        self.alpha = alpha
        self.gof = self.make_gof()

    def make_gof(self):
        if self.test_mode == 'cvm':
            gof = CVM()
        elif self.test_mode == 'ks':
            gof = KS()
        elif self.test_mode == 'ksd-u':
            gof = KSD(cfg['num_boostraps'], False)
        elif self.test_mode == 'ksd-v':
            gof = KSD(cfg['num_boostraps'], True)
        elif self.test_mode == 'mmd':
            gof = MMD(cfg['num_permutations'])
        else:
            raise ValueError('Not valid test mode')
        return gof

    def test(self, input):
        alter_noise = cfg['alter_noise']
        alter_num_samples = cfg['alter_num_samples']
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        print(null.size(), alter.size())
        alter = alter + alter_noise * torch.randn(alter.size(), device=alter.device)
        null_samples = torch.stack(torch.split(null, alter_num_samples, dim=0), dim=0)
        alter_samples = torch.stack(torch.split(alter, alter_num_samples, dim=0), dim=0)
        if self.test_mode in ['cvm', 'ks']:
            alter_samples = alter_samples.cpu().numpy()
            null_model = eval('models.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic, pvalue = self.gof.test(alter_samples, null_model)
        elif self.test_mode in ['ksd-u', 'ksd-v']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_model = eval('models.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic, pvalue = self.gof.test(null_samples, alter_samples, null_model)
        elif self.test_mode in ['mmd']:
            null_samples = null_samples
            alter_samples = alter_samples
            statistic, pvalue = self.gof.test(null_samples, alter_samples)
        else:
            raise ValueError('Not valid test mode')
        print(statistic)
        print(pvalue)
        exit()
        return statistic, pvalue
