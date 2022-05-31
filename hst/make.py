import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--model', default=None, type=str)
parser.add_argument('--split_round', default=65535, type=int)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    world_size = args['world_size']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    mode = args['mode']
    data = args['data']
    model = args['model']
    split_round = args['split_round']
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in
               list(range(init_gpu, init_gpu + num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, mode, data)
    if mode == 'ptb':
        script_name = [['{}_gof.py'.format(run)]]
        if data == 'MVN':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            ptb = []
            ptb_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9, 0.95,
                        1, 2]
            ptb_logvar = float(0)
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_i = '{}-{}'.format(ptb_mean_i, ptb_logvar)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                          0.95, 1, 2]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}'.format(ptb_mean, ptb_logvar_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls = controls_mean + controls_logvar
        elif data == 'GMM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            ptb = []
            ptb_mean = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9, 0.95,
                        1, 2]
            ptb_logvar = float(0)
            ptb_logweight = float(0)
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_i = '{}-{}-{}'.format(ptb_mean_i, ptb_logvar, ptb_logweight)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                          0.95, 1, 2]
            ptb_logweight = float(0)
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}-{}'.format(ptb_mean, ptb_logvar_i, ptb_logweight)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = float(0)
            ptb_logweight = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                             0.95, 1, 2]
            for i in range(len(ptb_logweight)):
                ptb_logweight_i = float(ptb_logweight[i])
                ptb_i = '{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_logweight = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            controls = controls_mean + controls_logvar + controls_logweight
        elif data == 'RBM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'hst-b-g', 'hst-b-e']
            ptb = []
            ptb_W = [0, 0.005, 0.007, 0.009, 0.01, 0.011, 0.012, 0.014, 0.015, 0.016, 0.018, 0.02, 0.025, 0.03, 0.035,
                     0.04, 0.045, 0.05, 0.075, 0.1]
            for i in range(len(ptb_W)):
                ptb_W_i = float(ptb_W[i])
                ptb_i = '{}'.format(ptb_W_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_W = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                       control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'hst-b-g']
            ptb = []
            ptb_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0]
            for i in range(len(ptb_tau)):
                ptb_W_i = float(ptb_tau[i])
                ptb_i = '{}'.format(ptb_W_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_W = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                       control_name)
            controls = controls_W
        else:
            raise ValueError('not valid data')
    elif mode == 'ds':
        script_name = [['{}_gof.py'.format(run)]]
        if data == 'MVN':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls = controls_mean + controls_logvar
        elif data == 'GMM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            ptb_mean = float(0)
            ptb_logvar = float(0)
            ptb_logweight = float(1)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logweight = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            controls = controls_mean + controls_logvar + controls_logweight
        elif data == 'RBM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'hst-b-g', 'hst-b-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_W = float(0.03)
            ptb = ['{}'.format(ptb_W)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_W = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                       control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'hst-b-g']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_tau = float(1)
            ptb = ['{}'.format(ptb_tau)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_tau = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                         control_name)
            controls = controls_tau
        else:
            raise ValueError('Not valid data')
    elif mode == 'noise':
        script_name = [['{}_gof.py'.format(run)]]
        if data == 'MVN':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            noise = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
            noise = [str(float(x)) for x in noise]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            controls = controls_mean + controls_logvar
        elif data == 'GMM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            noise = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
            noise = [str(float(x)) for x in noise]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_mean = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                          control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logvar = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                            control_name)
            ptb_mean = float(0)
            ptb_logvar = float(0)
            ptb_logweight = float(1)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logweight = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                               control_name)
            controls = controls_mean + controls_logvar + controls_logweight
        elif data == 'RBM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'hst-b-g', 'hst-b-e']
            noise = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
            noise = [str(float(x)) for x in noise]
            ptb_W = float(0.03)
            ptb = ['{}'.format(ptb_W)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_W = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode,
                                       control_name)
            controls = controls_W
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    s = '#!/bin/bash\n'
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                run_file = open('./{}_{}.sh'.format(filename, k), 'w')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        run_file = open('./{}_{}.sh'.format(filename, k), 'w')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
