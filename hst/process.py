import copy
import itertools
import json
import numpy as np
import os
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_curve, auc

result_path = os.path.join('output', 'result')
save_format = 'pdf'
vis_path = os.path.join('output', 'vis', save_format)
num_experiments = 1
exp = [str(x) for x in list(range(num_experiments))]
dpi = 300


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(mode, data):
    if mode == 'ptb':
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
            controls_mean = make_controls(control_name)
            ptb = []
            ptb_mean = float(0)
            ptb_logvar = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                          0.95, 1, 2]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_i = '{}-{}'.format(ptb_mean, ptb_logvar_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_logvar = make_controls(control_name)
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
            controls_mean = make_controls(control_name)
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
            controls_logvar = make_controls(control_name)
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
            controls_logweight = make_controls(control_name)
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
            controls_W = make_controls(control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'hst-b-g']
            # test_mode = ['lrt-b-g', 'hst-b-g']
            ptb = []
            ptb_tau = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0]
            for i in range(len(ptb_tau)):
                ptb_tau_i = float(ptb_tau[i])
                ptb_i = '{}'.format(ptb_tau_i)
                ptb.append(ptb_i)
            control_name = [[[data], test_mode, ptb, ['100'], ['0']]]
            controls_tau = make_controls(control_name)
            controls = controls_tau
        else:
            raise ValueError('not valid data')
    elif mode == 'ds':
        if data == 'MVN':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_mean = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logvar = make_controls(control_name)
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
            controls_mean = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logvar = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(0)
            ptb_logweight = float(1)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_logweight = make_controls(control_name)
            controls = controls_mean + controls_logvar + controls_logweight
        elif data == 'RBM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'hst-b-g', 'hst-b-e']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_W = float(0.03)
            ptb = ['{}'.format(ptb_W)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_W = make_controls(control_name)
            controls = controls_W
        elif data == 'EXP':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'hst-b-g']
            data_size = [5, 10, 20, 30, 40, 50, 80, 150, 200]
            data_size = [str(int(x)) for x in data_size]
            ptb_tau = float(1)
            ptb = ['{}'.format(ptb_tau)]
            control_name = [[[data], test_mode, ptb, data_size, ['0']]]
            controls_tau = make_controls(control_name)
            controls = controls_tau
        else:
            raise ValueError('Not valid data')
    elif mode == 'noise':
        if data == 'MVN':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'lrt-b-g', 'lrt-b-e', 'hst-b-g', 'hst-b-e']
            noise = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
            noise = [str(float(x)) for x in noise]
            ptb_mean = float(1)
            ptb_logvar = float(0)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_mean = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb = ['{}-{}'.format(ptb_mean, ptb_logvar)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logvar = make_controls(control_name)
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
            controls_mean = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(1)
            ptb_logweight = float(0)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logvar = make_controls(control_name)
            ptb_mean = float(0)
            ptb_logvar = float(0)
            ptb_logweight = float(1)
            ptb = ['{}-{}-{}'.format(ptb_mean, ptb_logvar, ptb_logweight)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_logweight = make_controls(control_name)
            controls = controls_mean + controls_logvar + controls_logweight
        elif data == 'RBM':
            test_mode = ['ksd-u', 'ksd-v', 'mmd', 'hst-b-g', 'hst-b-e']
            noise = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10]
            noise = [str(float(x)) for x in noise]
            ptb_W = float(0.03)
            ptb = ['{}'.format(ptb_W)]
            control_name = [[[data], test_mode, ptb, ['100'], noise]]
            controls_W = make_controls(control_name)
            controls = controls_W
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    write = False
    # mode = ['ptb', 'ds', 'noise']
    mode = ['ptb', 'ds']
    # mode = ['ptb']
    # data_name = ['MVN', 'GMM', 'RBM']
    # mode = ['ptb']
    # data_name = ['MVN', 'RBM']
    data_name = ['EXP']
    controls = []
    for i in range(len(mode)):
        mode_i = mode[i]
        for j in range(len(data_name)):
            data_j = data_name[j]
            control_list = make_control_list(mode_i, data_j)
            controls = controls + control_list
    processed_result_exp, processed_result_history = process_result(controls)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    df_exp = make_df_result(extracted_processed_result_exp, 'exp', write)
    df_history = make_df_result(extracted_processed_result_history, 'history', write)
    # make_vis(df_history, 'ptb')
    # make_vis(df_history, 'ds')
    # make_vis_statistic(df_exp, 'ptb')
    make_vis_roc(df_history, 'ptb')
    return


def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history


def extract_result(control, model_tag, processed_result_exp, processed_result_history):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            for k in base_result['logger'].mean:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_exp[metric_name]['exp'][exp_idx] = base_result['logger'].mean[k]
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger'].history[k]
            metric_name = 't1-mean'
            if metric_name not in processed_result_exp:
                processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            t1 = np.array(base_result['gof'].statistic['t1'])
            valid_mask = ~(np.isinf(t1) | np.isnan(t1))
            t1 = t1[valid_mask]
            if len(t1) == 0:
                t1 = np.finfo(np.float32).min
            else:
                t1 = np.mean(t1).item()
            # if t1 < -191809:
            #     t1 = -19180.9
            processed_result_exp[metric_name]['exp'][exp_idx] = t1
            metric_name = 't1-std'
            if metric_name not in processed_result_exp:
                processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            t1 = np.array(base_result['gof'].statistic['t1'])
            valid_mask = ~(np.isinf(t1) | np.isnan(t1))
            t1 = t1[valid_mask]
            if len(t1) == 0:
                t1_std = 0
            else:
                t1_std = np.std(t1).item()
                # if t1_std > 1000000:
                #     t1_std = 110023
                # t1_std = t1_std / 10
            processed_result_exp[metric_name]['exp'][exp_idx] = t1_std
            metric_name = 't2-mean'
            if metric_name not in processed_result_exp:
                processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            t2 = np.array(base_result['gof'].statistic['t2'])
            valid_mask = ~(np.isinf(t2) | np.isnan(t2))
            t2 = t2[valid_mask]
            if len(t2) == 0:
                t2 = np.finfo(np.float32).max
            else:
                t2 = np.mean(t2).item()
            processed_result_exp[metric_name]['exp'][exp_idx] = t2
            metric_name = 't2-std'
            if metric_name not in processed_result_exp:
                processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
            t2 = np.array(base_result['gof'].statistic['t2'])
            valid_mask = ~(np.isinf(t2) | np.isnan(t2))
            t2 = t2[valid_mask]
            if len(t2) == 0:
                t2_std = 0
            else:
                t2_std = np.std(t2).item()
            processed_result_exp[metric_name]['exp'][exp_idx] = t2_std
            num_trials_roc = 100
            metric_name = 'fpr'
            if metric_name not in processed_result_history:
                processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
            metric_name = 'tpr'
            if metric_name not in processed_result_history:
                processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
            t1 = np.array(base_result['gof'].pvalue['t1'])[:num_trials_roc].reshape(-1)
            t2 = np.array(base_result['gof'].pvalue['t2'])[:num_trials_roc].reshape(-1)
            valid_mask = ~(np.isinf(t1) | np.isnan(t1) | np.isinf(t2) | np.isnan(t2))
            t1 = t1[valid_mask]
            t2 = t2[valid_mask]
            y_true = np.array([1] * len(t1) + [0] * len(t2))
            score_arr = np.append(t1, t2)
            fpr, tpr, _ = roc_curve(y_true, score_arr, pos_label=1)
            processed_result_history['fpr']['history'][exp_idx] = fpr
            processed_result_history['tpr']['history'][exp_idx] = tpr
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]])
    return


def summarize_result(processed_result):
    if 'exp' in processed_result:
        pivot = 'exp'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
        processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
        processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
        processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
        processed_result[pivot] = processed_result[pivot].tolist()
    elif 'history' in processed_result:
        pivot = 'history'
        processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
        processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
        processed_result['std'] = np.std(processed_result[pivot], axis=0)
        processed_result['max'] = np.max(processed_result[pivot], axis=0)
        processed_result['min'] = np.min(processed_result[pivot], axis=0)
        processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
        processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
        processed_result[pivot] = processed_result[pivot].tolist()
    else:
        for k, v in processed_result.items():
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result:
        exp_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if exp_name not in extracted_processed_result:
            extracted_processed_result[exp_name] = defaultdict()
        extracted_processed_result[exp_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[exp_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df_result(extracted_processed_result, mode_name, write):
    df = defaultdict(list)
    for exp_name in extracted_processed_result:
        for k in extracted_processed_result[exp_name]:
            exp_name_list = exp_name.split('_')
            ptb, alter_num_samples, alter_noise = exp_name_list[-3:]
            ptb_list = ptb.split('-')
            for i in range(len(ptb_list)):
                ptb_list_ = copy.deepcopy(ptb_list)
                index_name = [ptb_list[i]]
                ptb_list_[i] = 'x'
                ptb_i = '-'.join(ptb_list_)
                exp_name_ = '_'.join([*exp_name_list[:-3], ptb_i, *exp_name_list[-2:]])
                df_name = '{}_{}'.format(exp_name_, k)
                df[df_name].append(
                    pd.DataFrame(data=[extracted_processed_result[exp_name][k]], index=index_name))
            index_name = [alter_num_samples]
            alter_num_samples_ = 'x'
            exp_name_ = '_'.join([*exp_name_list[:-2], alter_num_samples_, *exp_name_list[-1:]])
            df_name = '{}_{}'.format(exp_name_, k)
            df[df_name].append(
                pd.DataFrame(data=[extracted_processed_result[exp_name][k]], index=index_name))
            index_name = [alter_noise]
            alter_noise_ = 'x'
            exp_name_ = '_'.join([*exp_name_list[:-1], alter_noise_])
            df_name = '{}_{}'.format(exp_name_, k)
            df[df_name].append(
                pd.DataFrame(data=[extracted_processed_result[exp_name][k]], index=index_name))
    if write:
        startrow = 0
        writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode_name), engine='xlsxwriter')
        for df_name in df:
            concat_df = pd.concat(df[df_name])
            if len(concat_df) > 1:
                df[df_name] = concat_df
                df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
                writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
                startrow = startrow + len(df[df_name].index) + 3
        writer.save()
    else:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
    return df


def make_vis(df, vis_mode):
    color_dict = {'ksd-u': 'blue', 'ksd-v': 'cyan', 'lrt-b-g': 'black', 'lrt-b-e': 'gray', 'hst-b-g': 'red',
                  'hst-b-e': 'orange', 'mmd': 'green'}
    linestyle_dict = {'ksd-u': '-', 'ksd-v': '--', 'lrt-b-g': '-', 'lrt-b-e': '--', 'hst-b-g': '-',
                      'hst-b-e': '--', 'mmd': '-'}
    label_dict = {'ksd-u': 'KSD-U', 'ksd-v': 'KSD-V', 'lrt-b-g': 'LRT (Simple)', 'lrt-b-e': 'LRT (Composite)',
                  'hst-b-g': 'HST (Simple)', 'hst-b-e': 'HST (Composite)', 'mmd': 'MMD'}
    marker_dict = {'ksd-u': 'X', 'ksd-v': 'x', 'lrt-b-g': 'D', 'lrt-b-e': 'd',
                   'hst-b-g': 'o', 'hst-b-e': '^', 'mmd': 's'}
    label_loc_dict = {'Power': 'lower right', 't1': 'lower right', 't2': 'lower right'}
    xlabel_dict = {'ptb': 'Perturbation Magnitude $\sigma_{ptb}$', 'ds': 'Sample Size $n$',
                   'noise': 'Noise Magnitude $\sigma_{s}$'}
    exp_xlabel_dict = {'ptb': '$\\tau_{ptb}$', 'ds': 'Sample Size $n$'}
    fontsize = {'legend': 14, 'label': 16, 'ticks': 16}
    figsize = (10, 4)
    capsize = 3
    capthick = 3
    capsize = None
    capthick = None
    fig = {}
    ax_dict_1, ax_dict_2 = {}, {}
    for df_name in df:
        df_name_list = df_name.split('_')
        ptb, alter_num_samples, alter_noise = df_name_list[2], df_name_list[3], df_name_list[4]
        metric_name, stats = df_name_list[-2], df_name_list[-1]
        condition = len(df_name_list) == 7 and len(df[df_name]) > 1 and metric_name == 'Power-t2' and stats == 'mean'
        if vis_mode == 'ptb':
            condition = condition and 'x' in ptb
        elif vis_mode == 'ds':
            condition = condition and 'x' in alter_num_samples
        elif vis_mode == 'noise':
            condition = condition and 'x' in alter_noise
        else:
            raise ValueError('Not valid mode')
        if condition:
            data_name = df_name_list[0]
            test_mode = df_name_list[1]
            df_name_t2 = df_name
            x_t2 = df[df_name_t2].index.values
            x_t2 = np.array([float(x_) for x_ in x_t2])
            sorted_idx_t2 = np.argsort(x_t2)
            x_t2 = x_t2[sorted_idx_t2]
            y_t2 = df[df_name_t2].to_numpy()
            y_t2 = y_t2[sorted_idx_t2]
            y_t2_mean = y_t2.mean(axis=-1)
            y_t2_err = y_t2.std(axis=-1)
            if vis_mode == 'ptb':
                x_t2 = x_t2[1:]
                y_t2_mean = y_t2_mean[1:]
                y_t2_err = y_t2_err[1:]
            df_name_t1 = '_'.join([*df_name_list[:-2], 'Power-t1', stats])
            x_t1 = df[df_name_t1].index.values
            x_t1 = np.array([float(x_) for x_ in x_t1])
            sorted_idx_t1 = np.argsort(x_t1)
            x_t1 = x_t1[sorted_idx_t1]
            y_t1 = df[df_name_t1].to_numpy()
            y_t1 = y_t1[sorted_idx_t1]
            y_t1_mean = y_t1.mean(axis=-1)
            y_t1_err = y_t1.std(axis=-1)
            if vis_mode == 'ptb':
                x_t1 = x_t1[1:]
                y_t1_mean = y_t1_mean[1:]
                y_t1_err = y_t1_err[1:]
            fig_name = '_'.join([df_name_list[0], *df_name_list[2:-2]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(121)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(122)
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            label = test_mode
            ax_1.errorbar(x_t2, y_t2_mean, yerr=y_t2_err, color=color_dict[label], linestyle=linestyle_dict[label],
                          label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
            if data_name == 'EXP':
                xlabel = exp_xlabel_dict[vis_mode]
            else:
                xlabel = xlabel_dict[vis_mode]
            ax_1.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_1.set_ylabel('Power', fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict['t1'], fontsize=fontsize['legend'])
            ax_2.errorbar(x_t1, y_t1_mean, yerr=y_t1_err, color=color_dict[label], linestyle=linestyle_dict[label],
                          label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
            ax_2.set_xlabel(xlabel, fontsize=fontsize['label'])
            ax_2.set_ylabel('Type I Error Rate', fontsize=fontsize['label'])
            ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        if vis_mode == 'ptb':
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            lim_1 = list(ax_1.get_xlim())
            lim_2 = list(ax_2.get_xlim())
            lim_1[0] = 0
            lim_2[0] = 0
            ax_1.set_xlim(lim_1)
            ax_2.set_xlim(lim_2)
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'power', vis_mode, *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_statistic(df, vis_mode):
    color_dict = {'Alternative': 'red', 'Null': 'blue'}
    linestyle_dict = {'Alternative': '-', 'Null': '--'}
    label_dict = {'Alternative': 'Alternative', 'Null': 'Null'}
    marker_dict = {'Alternative': 'o', 'Null': '^'}
    label_loc_dict = {'statistic': 'lower right'}
    xlabel_dict = {'ptb': 'Perturbation Magnitude $\sigma_{ptb}$', 'ds': 'Sample Size $n$',
                   'noise': 'Noise Magnitude $\sigma_{s}$'}
    fontsize = {'legend': 14, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    capsize = 3
    capthick = 3
    fig = {}
    ax_dict_1 = {}
    for df_name in df:
        df_name_list = df_name.split('_')
        ptb, alter_num_samples, alter_noise = df_name_list[2], df_name_list[3], df_name_list[4]
        metric_name, stats = df_name_list[-2], df_name_list[-1]
        condition = len(df_name_list) == 7 and len(df[df_name]) > 1 and metric_name == 't2-mean' and stats == 'mean'
        if vis_mode == 'ptb':
            condition = condition and 'x' in ptb
        elif vis_mode == 'ds':
            condition = condition and 'x' in alter_num_samples
        elif vis_mode == 'noise':
            condition = condition and 'x' in alter_noise
        else:
            raise ValueError('Not valid mode')
        if condition:
            df_name_t2 = df_name
            x_t2 = df[df_name_t2].index.values
            x_t2 = np.array([float(x_) for x_ in x_t2])
            sorted_idx_t2 = np.argsort(x_t2)
            x_t2 = x_t2[sorted_idx_t2]
            y_t2 = df[df_name_t2].to_numpy()
            y_t2_mean = y_t2[sorted_idx_t2].reshape(-1)

            df_name_t2_std = '_'.join([*df_name_list[:-2], 't2-std', stats])
            y_t2_err = df[df_name_t2_std].to_numpy()
            y_t2_err = y_t2_err[sorted_idx_t2].reshape(-1)

            df_name_t1 = '_'.join([*df_name_list[:-2], 't1-mean', stats])
            x_t1 = df[df_name_t1].index.values
            x_t1 = np.array([float(x_) for x_ in x_t1])
            sorted_idx_t1 = np.argsort(x_t1)
            x_t1 = x_t1[sorted_idx_t1]
            y_t1 = df[df_name_t1].to_numpy()
            y_t1_mean = y_t1[sorted_idx_t1].reshape(-1)

            df_name_t1_std = '_'.join([*df_name_list[:-2], 't1-std', stats])
            y_t1_err = df[df_name_t1_std].to_numpy()
            y_t1_err = y_t1_err[sorted_idx_t1].reshape(-1)

            fig_name = '_'.join([*df_name_list])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            label = 'Alternative'
            ax_1.errorbar(x_t2, y_t2_mean, yerr=y_t2_err, color=color_dict[label], linestyle=linestyle_dict[label],
                          label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
            label = 'Null'
            ax_1.errorbar(x_t1, y_t1_mean, yerr=y_t1_err, color=color_dict[label], linestyle=linestyle_dict[label],
                          label=label_dict[label], marker=marker_dict[label], capsize=capsize, capthick=capthick)
            ax_1.set_xlabel(xlabel_dict[vis_mode], fontsize=fontsize['label'])
            ax_1.set_ylabel('Test Statistic', fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.set_yticks([0])
            ax_1.legend(loc=label_loc_dict['statistic'], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_1 = ax_dict_1[fig_name]
        lim_1 = list(ax_1.get_ylim())
        if lim_1[0] > 0:
            lim_1[0] = 0
        ax_1.set_ylim(lim_1)
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'statistic', vis_mode, *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_roc(df, vis_mode):
    color_dict = {'ksd-u': 'blue', 'ksd-v': 'cyan', 'lrt-b-g': 'black', 'lrt-b-e': 'gray', 'hst-b-g': 'red',
                  'hst-b-e': 'orange', 'mmd': 'green'}
    linestyle_dict = {'ksd-u': '-', 'ksd-v': '--', 'lrt-b-g': '-', 'lrt-b-e': '--', 'hst-b-g': '-',
                      'hst-b-e': '--', 'mmd': '-'}
    label_dict = {'ksd-u': 'KSD-U', 'ksd-v': 'KSD-V', 'lrt-b-g': 'LRT (Simple)', 'lrt-b-e': 'LRT (Composite)',
                  'hst-b-g': 'HST (Simple)', 'hst-b-e': 'HST (Composite)', 'mmd': 'MMD'}
    marker_dict = {'ksd-u': 'X', 'ksd-v': 'x', 'lrt-b-g': 'D', 'lrt-b-e': 'd',
                   'hst-b-g': 'o', 'hst-b-e': '^', 'mmd': 's'}
    label_loc_dict = {'statistic': 'lower right'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    ptb_pivot_ptb_dict = {'MVN-x-0.0': '0.5', 'MVN-0.0-x': '0.5', 'GMM-x-0.0-0.0': '0.5', 'GMM-0.0-x-0.0': '0.5',
                          'GMM-0.0-0.0-x': '0.5', 'RBM-x': '0.01', 'EXP-x': '1.0'}
    ds_pivot_ptb_dict = {'MVN-x': '50', 'GMM-x': '50', 'RBM-x': '50', 'EXP-x': '50'}
    for df_name in df:
        df_name_list = df_name.split('_')
        ptb, alter_num_samples, alter_noise = df_name_list[2], df_name_list[3], df_name_list[4]
        metric_name, stats = df_name_list[-2], df_name_list[-1]
        condition = len(df_name_list) == 7 and len(df[df_name]) > 1 and metric_name == 'fpr' and stats == 'mean'
        if vis_mode == 'ptb':
            condition = condition and 'x' in ptb
        elif vis_mode == 'ds':
            condition = condition and 'x' in alter_num_samples
        elif vis_mode == 'noise':
            condition = condition and 'x' in alter_noise
        else:
            raise ValueError('Not valid mode')
        if condition:
            test_mode = df_name_list[1]
            if vis_mode == 'ptb':
                pivot_index = ptb_pivot_ptb_dict['{}-{}'.format(df_name_list[0], df_name_list[2])]
            elif vis_mode == 'ds':
                pivot_index = ds_pivot_ptb_dict['{}-{}'.format(df_name_list[0], df_name_list[3])]
            else:
                raise ValueError('Not valid mode')
            df_name_fpr = df_name
            fpr = df[df_name_fpr].loc[pivot_index].to_numpy().reshape(-1)
            fpr = fpr[~np.isnan(fpr)]
            df_name_tpr = '_'.join([*df_name_list[:-2], 'tpr', stats])
            tpr = df[df_name_tpr].loc[pivot_index].to_numpy().reshape(-1)
            tpr = tpr[~np.isnan(tpr)]
            auc_ = auc(fpr, tpr)
            fig_name = '_'.join([df_name_list[0], *df_name_list[2:-2]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            label = test_mode
            ax_1.plot(fpr, tpr, color=color_dict[label], linestyle=linestyle_dict[label],
                      label='{}, AUC={:.2f}'.format(label_dict[label], auc_), marker=marker_dict[label])
            ax_1.set_xlabel('False Positive Rate', fontsize=fontsize['label'])
            ax_1.set_ylabel('True Positive Rate', fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict['statistic'], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_1 = ax_dict_1[fig_name]
        plt.plot([0, 1], [0, 1], linestyle="--", color='lightgray')
        # ax_1.set_xlim([0, 1])
        # ax_1.set_ylim([0.0, 1.05])
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'roc', vis_mode, *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
