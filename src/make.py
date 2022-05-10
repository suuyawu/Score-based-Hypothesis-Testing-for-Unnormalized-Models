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
    filename = '{}_{}_{}_{}'.format(run, mode, data, model)
    if mode == 'teacher':
        script_name = [['{}_teacher.py'.format(run)]]
        control_name = [[[data], [model], ['1'], ['iid']]]
        controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        if data not in ['CIFAR100']:
            control_name = [[[data], [model], ['10'], ['iid', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3']]]
        else:
            control_name = [[[data], [model], ['10'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3']]]
        controls_10 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls = controls_1 + controls_10
    elif mode == 'uhc':
        script_name = [['{}_student_s.py'.format(run)]]
        control_name = [[[data], [model], ['1'], ['iid'], [data], [model], ['uhc']]]
        controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        if data not in ['CIFAR100']:
            control_name = [
                [[data], [model], ['10'], ['iid', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], [data], [model],
                 ['uhc']]]
        else:
            control_name = [
                [[data], [model], ['10'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3'], [data], [model], ['uhc']]]
        controls_10 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls = controls_1 + controls_10
    elif mode == 'dist':
        script_name = [['{}_student_s.py'.format(run)]]
        control_name = [[[data], [model], ['1'], ['iid'], [data], [model], ['dist']]]
        controls_1 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        if data not in ['CIFAR100']:
            control_name = [
                [[data], [model], ['10'], ['iid', 'non-iid-l-2', 'non-iid-d-0.1', 'non-iid-d-0.3'], [data], [model],
                 ['dist']]]
        else:
            control_name = [
                [[data], [model], ['10'], ['iid', 'non-iid-d-0.1', 'non-iid-d-0.3'], [data], [model], ['dist']]]
        controls_10 = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, control_name)
        controls = controls_1 + controls_10
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
