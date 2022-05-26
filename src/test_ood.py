import argparse
import datetime
import os

os.environ['OMP_NUM_THREADS'] = '1'
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from utils import save, load, to_device, process_control, resume, collate, process_dataset
from logger import make_logger
from modules import OutofDistributionDetection

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        null_model_tag_list = [str(seeds[i]), '_'.join(cfg['control_name'].split('_')[:1])]
        cfg['null_model_tag'] = '_'.join([x for x in null_model_tag_list if x])
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    process_control()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    dataset = {'test': dataset['test']}
    dataset = split_dataset(dataset)
    model = eval('models.{}(cfg["rbm"]).to(cfg["device"])'.format(cfg['model_name']))
    result = resume('./output/model/{}_{}.pt'.format(cfg['null_model_tag'], 'best'))
    model.load_state_dict(result['model_state_dict'])
    ood = OutofDistributionDetection(cfg['test_mode'])
    metric = Metric(cfg['data_name'], {'test': ['AUCROC']})
    logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    for i in range(len(dataset)):
        make_hs(dataset[i], model, ood, i)
    for i in range(len(dataset)):
        test(ood, metric, logger, i)
        logger.reset()
    result = {'cfg': cfg, 'logger': logger, 'ood': ood}
    save(result, os.path.join('output', 'result', '{}.pt'.format(cfg['model_tag'])))
    return


def make_hs(dataset, null_model, ood, label_idx):
    with torch.no_grad():
        data_loader = make_data_loader(dataset, 'ood', shuffle={'test': True},
                                       batch_size={'test': cfg['alter_num_samples']}, drop_last={'test': True})
        for i, input in enumerate(data_loader['test']):
            input = collate(input)
            input = to_device(input, cfg['device'])
            output = ood.detect(input, null_model)
            ood.update(output, label_idx)
    return


def test(ood, metric, logger, label_idx):
    logger.safe(True)
    with torch.no_grad():
        t1 = np.array(ood.hs[cfg['null_label']])
        t2 = np.array(ood.hs[label_idx])
        min_length = min(len(t1), len(t2))
        print(label_idx, len(t1), len(t2), min_length)
        t1 = t1[:min_length]
        t2 = t2[:min_length]
        input = {'target': np.append(np.zeros(min_length), np.ones(min_length))}
        output = {'target': np.append(t1, t2)}
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', 1)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Label: {}/{}'.format(label_idx, cfg['target_size'] - 1)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()
