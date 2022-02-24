# HScore for OOD Detection
score matching out-of-distribution detection

1. "./pretrained_score_function" This includes implementation for the score-based test statistics on CIFAR10 (in-distribution) and SVHN (out-of-distribution).
	- "score_sde_pytorch/test_st.py" is the main file for implementation
	- "score_sde_pytorch/utils_st.py" calculates the test statistic based on pretrained score function
	- "output/results1.pdf" demonstrates the huge gap between statistics for in-distribution data and for out-of distribution data.
