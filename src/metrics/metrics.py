import torch
import torch.nn.functional as F
from config import cfg


def Power(pvalue, alpha):
    pvalue = torch.tensor(pvalue)
    power = (pvalue < alpha).float().mean().item()
    return power


class Metric(object):
    def __init__(self, data_name, metric_name):
        self.data_name = data_name
        self.metric_name = metric_name
        self.metric = {'Power': (lambda input, output: Power(output['pvalue'], cfg['alpha']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation
