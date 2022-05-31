# Hyvarinen Score Difference Test for CIFAR-10 versus Tiny-ImageNet
This file contains a PyTorch implementation of the heuristic HST for Out-of-Distribution (OOD) detection
The code for implementation is heavily built on the official code for Score-Based Generative Modeling through Stochastic Differential Equations (https://github.com/yang-song/score_sde).

## How To Run
Perform the heuristic HST for Out-of-Distribution (OOD) detection through `hst.py`.

To perform OOD on CIFAR-10(in-distribution) versus Tiny-ImageNet (out-of-distribution):
```
python hst.py
```
