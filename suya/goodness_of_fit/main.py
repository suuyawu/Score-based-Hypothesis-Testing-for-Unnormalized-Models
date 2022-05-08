"""goodness of fit test simulations
"""
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from benchmarks import Distance_test_batch, Nonparametric_test_batch
from hscore_test import Density_test_batch
from distribution import GaussianMixture, GaussBernRBM
from utils import Sample_Dataset

training = True

def simulation(eval_name, dist_name, test_names, params, alter_ptb_name, num_tests):
    if eval_name == 'alternative_shift':
        if dist_name == '1dgmm':
            eval_levels = alter_ptd_std = np.array([0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7, 1, 1.5, 2, 2.5, 3])
        if dist_name == 'gbrbm':
            eval_levels = alter_ptd_std = np.array([0,0.005,0.007,0.009,0.01,0.011,0.012,0.013,0.014,0.016,0.018,0.02, 0.025, 0.03, 0.035])
        sample_size = np.array([100]*len(alter_ptd_std))
        data_ptd_std = np.array([0.]*len(alter_ptd_std))
    if eval_name == 'sample_size':
        eval_levels = sample_size = np.array([5, 10, 20, 30, 50, 70, 90, 120, 150, 200])
        alter_ptd_std = np.array([1]*len(sample_size))
        data_ptd_std = np.array([0.]*len(sample_size))
    if eval_name == 'noisy_data':
        eval_levels = data_ptd_std = np.array([0, 0.005, 0.007, 0.009, 0.012, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.5, 0.7, 1])
        alter_ptd_std = np.array([1]*len(data_ptd_std))
        sample_size = np.array([100]*len(data_ptd_std))
    powers = {}
    times = {}
    for name in test_names:
        powers.update({name:[]})
        times.update({name:[]})
    null_params = {}
    for key in params.keys():
        null_params.update({key:params[key].detach().clone()})
    params_ptb = params.copy()
    
    for i, (num_samples, eps_a, eps_d) in enumerate(zip(sample_size, alter_ptd_std, data_ptd_std)):
        print('[Evaluation Setup] Sample size: {}; Alternative model perturbation level: {}; Data perturbation level {}'.format(num_samples, eps_a, eps_d))
        if dist_name == '1dgmm':
            model0 = GaussianMixture(null_params)
            # params_ptb[alter_ptb_name] = params_ptb[alter_ptb_name]+eps_a*(torch.empty(params_ptb[alter_ptb_name].shape).uniform_(0, 1)*2.-1.)
            params_ptb[alter_ptb_name] = params_ptb[alter_ptb_name]+eps_a*torch.randn_like(params_ptb[alter_ptb_name])
            if alter_ptb_name == 'logweight':
                #normalize the weight vector
                params_ptb['logweight'][1] = torch.log(1- torch.exp(params_ptb['logweight'][0]))
            model1_true = GaussianMixture(params_ptb)
        elif dist_name == 'gbrbm':
            model0 = GaussBernRBM(null_params)
            params_ptb[alter_ptb_name] = params_ptb[alter_ptb_name]+eps_a*torch.randn_like(params_ptb[alter_ptb_name])
            model1_true = GaussBernRBM(params_ptb)

        null_samples = model0.sample(num_samples*10)
        alter_samples = model1_true.sample(num_samples*num_tests)
        alter_samples = alter_samples+ eps_d*torch.randn_like(alter_samples)
        alter_samples = alter_samples.view(num_tests, num_samples, -1)
        
        if training:
            model1 = None
            training_args = {
                'train_epoch': 50,
                'params': params,
                'params_ptb': params_ptb, #this is just for checking the training performance
                'dist_name': dist_name,
                'alter_ptb_name': alter_ptb_name,
            }
            print('need training')
        else:
            model1 = model1_true
            training_args = None
    
        #Test goodness-of-fit
        if 'Kolmogorov Smirnov' in test_names:
            result_ks = Nonparametric_test_batch('Kolmogorov Smirnov', alter_samples, model0)
            times['Kolmogorov Smirnov'].append(result_ks['time'])
            powers['Kolmogorov Smirnov'].append(result_ks['power'])
        if 'Cramér-von Mises' in test_names:
            result_cm = Nonparametric_test_batch('Cramér-von Mises', alter_samples, model0)
            times['Cramér-von Mises'].append(result_ks['time'])
            powers['Cramér-von Mises'].append(result_ks['power'])
        if 'LikelihoodRatio Chi2' in test_names:
            result_lrt = Density_test_batch('LikelihoodRatio Chi2', null_samples, alter_samples, model0, alter_model = model1, bootstrap_approx=False)
            times['LikelihoodRatio Chi2'].append(result_lrt['time'])
            powers['LikelihoodRatio Chi2'].append(result_lrt['power'])
        if 'LikelihoodRatio Bootstrap' in test_names:
            result_lrt = Density_test_batch('LikelihoodRatio Bootstrap', null_samples, alter_samples, model0, alter_model = model1, bootstrap_approx=True)
            times['LikelihoodRatio Bootstrap'].append(result_lrt['time'])
            powers['LikelihoodRatio Bootstrap'].append(result_lrt['power'])
        if 'HScore' in test_names:
            result_hscore = Density_test_batch('HScore', null_samples, alter_samples, model0, alter_model = model1, training_args=training_args)
            times['HScore'].append(result_hscore['time'])
            powers['HScore'].append(result_hscore['power'])
        if 'KSD-U' in test_names:
            result_ksd = Distance_test_batch('KSD-U', null_samples, alter_samples, model0)
            times['KSD-U'].append(result_ksd['time'])
            powers['KSD-U'].append(result_ksd['power'])
        if 'MMD' in test_names:
            result_ksd = Distance_test_batch('MMD', null_samples, alter_samples, None)
            times['MMD'].append(result_ksd['time'])
            powers['MMD'].append(result_ksd['power'])
        for key in powers.keys():
            print(key, powers[key])
    return eval_levels, powers, times

def experiments_RBM(num_trials):
    """Goodness of fit tests for RBM model weights shifts
    """
    xdim = 50
    hdim = 40
    params = {
        'W':torch.randn(xdim, hdim, dtype=torch.float64),
        'bx':torch.randn(xdim, dtype=torch.float64),
        'bh':torch.randn(hdim, dtype=torch.float64)
    }
    alter_ptb_name = ['W']
    test_names = ['HScore', 'KSD-U']
    trial_powers = {}
    for name in test_names:
        trial_powers.update({name:[]})

    for i in tqdm(range(num_trials)):
        eval_levels, powers, times = simulation(
            eval_name = 'alternative_shift', 
            dist_name = 'gbrbm', 
            test_names = ['HScore', 'KSD-U'], 
            params = params, 
            alter_ptb_name = alter_ptb_name[0],
            num_tests = 1000)
        for name in test_names:
            if i == 0:
                    trial_powers[name] = np.array(powers[name])
            else:
                trial_powers[name] = trial_powers[name]+ np.array(powers[name])
    for name in test_names:
        trial_powers[name] = trial_powers[name]/num_trials
    chars = ['-rD', '-go', '-bx', '-mp']
    for i, name in enumerate(test_names):
        plt.plot(eval_levels, np.array(trial_powers[name]), chars[i], label = '{} test'.format(name))
    plt.legend()
    plt.title('Power comparision with fixed test size 0.05')
    plt.savefig('../output/power_{}_{}.png'.format('gbgmm', alter_ptb_name[0]))
    plt.close()

def experiments_1dGMM(num_trials):
    """Goodness of fit tests for One-dimensional Gaussian mixture model mean and covariance shifts
    """
    params = {
        'logweight':torch.log(torch.tensor([0.2, 0.8])),
        'mean':torch.tensor([0., 5.]),
        'logvar':torch.tensor([0., 0.])
    }
    test_names = ['HScore', 'LikelihoodRatio Bootstrap', 'LikelihoodRatio Chi2', 'KSD-U', 'MMD']
    alter_ptb_names = ['mean', 'logvar']
    trial_powers = {}
    for name in test_names:
        trial_powers.update({name:[]})
    for alter_ptb_name in alter_ptb_names:
        for i in range(num_trials):
            eval_levels, powers, times = simulation(
                                            eval_name = 'alternative_shift', 
                                            dist_name = '1dgmm', 
                                            test_names = test_names, 
                                            params = params, 
                                            alter_ptb_name = alter_ptb_name,
                                            num_tests = 1000)
            for name in test_names:
                if i == 0:
                        trial_powers[name] = np.array(powers[name])
                else:
                    trial_powers[name] = trial_powers[name]+ np.array(powers[name])  
        for name in test_names:
            trial_powers[name] = trial_powers[name]/num_trials
        chars = ['-rD', '--go', '--gx', '-.bx', '-.mp']
        for i, name in enumerate(test_names):
            plt.plot(eval_levels, np.array(trial_powers[name]), chars[i], label = '{} test'.format(name))
        plt.legend()
        plt.title('Power comparision with fixed test size 0.05')
        plt.savefig('../output/power_{}_{}.png'.format('1dgmm', alter_ptb_name))
        plt.close()

if __name__ == '__main__':
    experiments_RBM(10)