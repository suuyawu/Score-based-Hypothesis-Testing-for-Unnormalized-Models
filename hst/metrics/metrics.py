import torch
import torch.nn.functional as F
from config import cfg
from sklearn.metrics import roc_auc_score


def Power(pvalue, alpha):
    pvalue = torch.tensor(pvalue)
    power = (pvalue < alpha).float().mean().item()
    return power


def AUCROC(output, target):
    aucroc = roc_auc_score(target, output)
    return aucroc


class Metric(object):
    def __init__(self, data_name, metric_name):
        self.data_name = data_name
        self.metric_name = metric_name
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'Power-t1': (lambda input, output: Power(output['pvalue_t1'], cfg['alpha'])),
                       'Power-t2': (lambda input, output: Power(output['pvalue_t2'], cfg['alpha'])),
                       'AUCROC': (lambda input, output: AUCROC(output['target'], input['target']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def make_pivot(self):
        if cfg['data_name'] in ['KDDCUP99']:
            pivot = float('inf')
            pivot_direction = 'down'
            pivot_name = 'Loss'
        else:
            pivot = None
            pivot_name = None
            pivot_direction = None
        return pivot, pivot_name, pivot_direction

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
