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

null_label = 4


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(mode, data):
    if mode == 'ood':
        if data == 'KDDCUP99':
            test_mode = ['hst']
            # control_name = [[[data], test_mode, ['none'], ['1', '5', '10', '15', '20']]]
            control_name = [[[data], test_mode, ['none'], ['1', '2', '4', '8', '10']]]
            controls = make_controls(control_name)
        else:
            raise ValueError('Not valid data')
    else:
        raise ValueError('Not valid mode')
    return controls


def main():
    write = False
    data_name = ['KDDCUP99']
    mode = ['ood']
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
    make_vis_ood(df_history, 'class')
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
            for k in base_result['logger'].history:
                metric_name = k.split('/')[1]
                if metric_name not in processed_result_exp:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_history[metric_name]['history'][exp_idx] = base_result['logger'].history[k]
            hs = base_result['ood'].hs
            t1 = np.array(hs[null_label])
            for i in range(len(hs)):
                metric_name = 'hs-{}'.format(i)
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                processed_result_history['hs-{}'.format(i)]['history'][exp_idx] = np.array(hs[i])
                t2 = np.array(hs[i])
                metric_name = 'fpr-{}'.format(i)
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                metric_name = 'tpr-{}'.format(i)
                if metric_name not in processed_result_history:
                    processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                y_true = np.array([0] * len(t1) + [1] * len(t2))
                score_arr = np.append(t1, t2)
                fpr, tpr, _ = roc_curve(y_true, score_arr, pos_label=1)
                processed_result_history['fpr-{}'.format(i)]['history'][exp_idx] = fpr
                processed_result_history['tpr-{}'.format(i)]['history'][exp_idx] = tpr
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
            alter_num_samples = exp_name_list[-1]
            index_name = [alter_num_samples]
            alter_num_samples_ = 'x'
            exp_name_ = '_'.join([*exp_name_list[:-1], alter_num_samples_])
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


def make_vis_ood(df, vis_mode):
    label_list = ['back', 'ipsweep', 'neptune', 'nmap', 'normal',
                  'pod', 'portsweep', 'satan', 'smurf', 'teardrop', 'unknown',
                  'warezclient']
    color_dict = {'1': 'blue', '2': 'orange', '4': 'green', '8': 'red', '10': 'purple'}
    linestyle_dict = {'1': '-', '2': '-', '4': '-', '8': '-', '10': '-'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    figsize = (15, 4)
    fig = {}
    ax_dict_1, ax_dict_2, ax_dict_3 = {}, {}, {}
    for df_name in df:
        df_name_list = df_name.split('_')
        metric_name, stats = df_name_list[-2], df_name_list[-1]
        condition = len(df_name_list) == 6 and 'fpr' in metric_name and stats == 'mean'
        if condition:
            metric_name, label_idx = metric_name.split('-')
            df_name_fpr = df_name
            fpr = df[df_name_fpr]
            df_name_tpr = '_'.join([*df_name_list[:-2], 'tpr-{}'.format(label_idx), stats])
            tpr = df[df_name_tpr]
            fig_name = '_'.join([*df_name_list[:-2], label_list[int(label_idx)]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(131)
                ax_dict_2[fig_name] = fig[fig_name].add_subplot(132)
                ax_dict_3[fig_name] = fig[fig_name].add_subplot(133)
            ax_1 = ax_dict_1[fig_name]
            ax_2 = ax_dict_2[fig_name]
            ax_3 = ax_dict_3[fig_name]
            for i in range(len(fpr)):
                fpr_i = fpr.iloc[i]
                num_alter_samples = fpr_i.name
                fpr_i = fpr_i[~np.isnan(fpr_i)]
                tpr_i = tpr.iloc[i]
                tpr_i = tpr_i[~np.isnan(tpr_i)]
                auc_i = auc(fpr_i, tpr_i)
                label = num_alter_samples
                ax_1.plot(fpr_i, tpr_i, color=color_dict[label], linestyle=linestyle_dict[label],
                          label='{}, AUC={:.2f}'.format('$n={}$'.format(label), auc_i))
                ax_1.set_xlabel('False Positive Rate', fontsize=fontsize['label'])
                ax_1.set_ylabel('True Positive Rate', fontsize=fontsize['label'])
                ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax_1.legend(loc='lower right', fontsize=fontsize['legend'])
                ax_1.set_title('(a) ROC', fontsize=fontsize['label'])
                if num_alter_samples == '1':
                    df_name_t1_1 = '_'.join([*df_name_list[:-3], 'x', 'hs-{}'.format(null_label), stats])
                    t1_1 = df[df_name_t1_1].loc[num_alter_samples]
                    df_name_t2_1 = '_'.join([*df_name_list[:-3], 'x', 'hs-{}'.format(label_idx), stats])
                    t2_1 = df[df_name_t2_1].loc[num_alter_samples]
                    ax_2.hist(t1_1, label=label_list[null_label], alpha=0.8, bins=20)
                    ax_2.hist(t2_1, label=label_list[int(label_idx)], alpha=0.8, bins=20)
                    ax_2.set_xlabel('Test Statistic', fontsize=fontsize['label'])
                    ax_2.set_ylabel('Frequency', fontsize=fontsize['label'])
                    ax_2.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax_2.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax_2.set_title('(b) $n=1$', fontsize=fontsize['label'])
                if num_alter_samples == '10':
                    df_name_t1_15 = '_'.join([*df_name_list[:-3], 'x', 'hs-{}'.format(null_label), stats])
                    t1_15 = df[df_name_t1_15].loc[num_alter_samples]
                    df_name_t2_15 = '_'.join([*df_name_list[:-3], 'x', 'hs-{}'.format(label_idx), stats])
                    t2_15 = df[df_name_t2_15].loc[num_alter_samples]
                    ax_3.hist(t1_15, label=label_list[null_label], alpha=0.8, bins=20)
                    ax_3.hist(t2_15, label=label_list[int(label_idx)], alpha=0.8, bins=20)
                    ax_3.set_xlabel('Test Statistic', fontsize=fontsize['label'])
                    ax_3.set_ylabel('Frequency', fontsize=fontsize['label'])
                    ax_3.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax_3.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                    ax_3.set_title('(c) $n=10$', fontsize=fontsize['label'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_2[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_dict_3[fig_name].grid(linestyle='--', linewidth='0.5')
        ax_1 = ax_dict_1[fig_name]
        ax_1.plot([0, 1], [0, 1], linestyle="--", color='lightgray')
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
