import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_control, resume, collate
from logger import make_logger
from modules import GoodnessOfFit

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    null_params = make_null_params(cfg['data_name'], cfg['ptb'])
    dataset = fetch_dataset(cfg['data_name'], null_params)
    data_loader = make_data_loader(dataset, 'ht')
    gof = GoodnessOfFit(cfg['test_mode'], cfg['alter_num_samples'], cfg['alter_noise'])
    metric = Metric(cfg['data_name'], {'test': ['Power']})
    logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], gof, metric, logger)
    result = {'cfg': cfg, 'logger': logger}
    save(result, os.path.join('output', 'result', '{}.pt'.format(cfg['model_tag'])))
    return


def make_null_params(data_name, ptb):
    null_params = load(os.path.join('output', 'null', 'params.pkl'))
    null_params = null_params[data_name][ptb]
    return null_params


def test(data_loader, gof, metric, logger):
    for i, input in enumerate(data_loader):
        logger.safe(True)
        input = collate(input)
        input = to_device(input, cfg['device'])
        output = gof.test(input)
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', 1)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Iter: {}/{}'.format(i, len(data_loader))]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        logger.safe(False)
    return


if __name__ == "__main__":
    main()
