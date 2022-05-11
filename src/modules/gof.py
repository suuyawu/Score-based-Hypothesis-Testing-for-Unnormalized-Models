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
from modules import CVM, KS, KSD, MMD, LRT, HST


class HypothesisTest:
    def __init__(self, test_mode, alter_num_samples, alter_noise, alpha):
        self.test_mode = test_mode
        self.alter_num_samples = alter_num_samples
        self.alter_noise = alter_noise
        self.alpha =alpha
        self.ht = self.make_ht(test_mode)

    def make_ht(self, test_mode):
        self.test_mode_dict = {'cvm': CVM, 'ks': KS}
        ht = self.test_mode_dict[test_mode]()
        return ht

    def test(self, input):
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        result = self.ht.test(input)
        return result
