# Score-based Hypothesis Testing for Unnormalized Models
This is an implementation of [Score-based Hypothesis Testing for Unnormalized Models](https://ieeexplore.ieee.org/document/9813688)
 
## Requirements
See `hst/requirements.txt` and `ood/requirements.txt`

## Instruction
- Perform Goodness of Fit Test through `hst/test_gof.py`.
     - Global hyperparameters are configured in config.yml
     - Experimental setup are listed in make.py 
     - Hyperparameters can be found at process_control() in utils.py
- Perform the heuristic HST for Out-of-Distribution (OOD) detection on Network Intrusion Data through `hst/test_ood.py`.
- Perform the heuristic HST for Out-of-Distribution (OOD) detection on Image Data through `ood/test_ood.py`. Our implementation is based on the official code for "Score-Based Generative Modeling through Stochastic Differential Equations: (https://github.com/yang-song/score_sde).



## Examples
 - Prepare datasets for Goodness of Fit Test
    ```ruby
    python hst/make_datasets.py
    ```
 - Generate scripts to perform Goodness of Fit Test with multivariate Normal distribution by varying the perturbation level of mean and logvar
    ```ruby
    python hst/make.py --mode ptb --data MVN --run test
    ```
 - Generate scripts to perform Goodness of Fit Test with Gauss-RBM by varing the sample size n
    ```ruby
    python hst/make.py --mode ds --data RBM --run test
    ```
 - Perform KSD-U test with multivariate Normal distribution at perburbation level of 0.1 on mean and 0.0 on logvar and sample size n=100
    ```ruby
    python hst/test_gof.py --control_name MVN_ksd-u_0.1-0.0_100_0
    ```
 - Perform HST (Simple) test with Exponential family at perburbation level 0.1 on tau and sample size n=100
    ```ruby
    python hst/test_gof.py --control_name EXP_hst-b-g_0.1_100_0
    ```
 - Perform HST (Composite) test with Gauss-RBM at perburbation level 1.0 on W and sample size n=30
    ```ruby
    python hst/test_gof.py --control_name RBM_hst-b-e_1.0_30_0
    ```
 - Perform computation comparison between LRT (Simple) and HST (Simple)
    ```ruby
    python hst/run_exp_time.py
    ```
 - Perform OOD Detection on Network Intrusion Data (n=1)
    ```ruby
    python hst/train_ood.py --control_name KDDCUP99_rbm
    python hst/test_ood.py --control_name KDDCUP99_hst_none_1
    ```
 - Perform OOD Detection on CIFAR-10(in-distribution) versus Tiny-ImageNet (out-of-distribution)
    ```ruby
    python ood/test_ood.py
    ```

## Results
- Multivariate Normal Distribution, power comparison by changing the perturbation level of $\log(\Sigma)$ $(n=100)$

![MVN_0.0-x_100_0](/asset/MVN_0.0-x_100_0.png)

- Gauss-Bernoulli RBM, Power comparison of various tests varying sample size and the perturbation level $\sigma_{ptb} = 0.03$

![RBM_0.03_x_0](/asset/RBM_0.03_x_0.png)

- (a) ROC curves and (b, c) histograms of test statistics of HST for OOD Detection on CIFAR10 (in-distribution) and Tiny ImageNet datasets (out-distribution)

![ood](/asset/ood.png)

## Acknowledgement
*Suya Wu  
Enmao Diao  
Khalil Elkhalil  
Jie Ding  
Vahid Tarokh*
