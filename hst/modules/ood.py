import torch
import models
from config import cfg


class OutofDistributionDetection:
    def __init__(self, test_mode):
        self.test_mode = test_mode
        self.hs = {i: [] for i in range(cfg['target_size'])}

    def detect(self, input, null_model):
        data_samples = input['data']
        if self.test_mode in ['hst']:
            hs = null_model.hscore(data_samples).sum().item()
        else:
            raise ValueError('Not valid test mode')
        output = {'hs': hs}
        return output

    def update(self, output, i):
        self.hs[i].append(output['hs'])
        return
