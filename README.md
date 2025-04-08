# Thesis

This repository contains all relevant code for my Masters thesis at Tilburg University : Msc in Cognitive Science & Artificial Intelligence. 

The main package developed for the project is `eeglearn` which is inside the `eeg-graph-learning` directory. 

## Project overview
### TL;DR
Self-supervised pre-training is the engine that powers all of the latest generations of language models. Here, the training objective is next token prediction. 

Similarly, this method has been attempted for EEG data. Furthermore, researchers have attempted to incorporate information about the placement of the EEG electrodes into the training process with Graph Neural networks. 

### Goal of the project

![alt text](https://github.com/onedeeper/thesis/blob/main/eeg_pretraining.png?raw=true)

Testing self-supervised pre-training on pretext tasks with graph neural networks [1-5] on a benchmark dataset [6].

### Personal goal
**My personal goal is to make the code I wrote for my Master's thesis better than the code I wrote for bachelor's thesis** ( [the data extraction package](https://github.com/onedeeper/observer) and [the analysis code](https://osf.io/mwn6e/files/osfstorage?view_only=)). 

To that end, I have tried very hard to:
1. Incorporate software engineering best principles (as far as I can learn them from disparate sources online or from asking an LLM). 
2. Make the code reproducible to the best of my ability ( [more info on how I tried  to do that](udesh.io/reproduce.html ) )

**I also focused on improving on the work I was doing week  after week**. This means I incorporated good ideas as I encountered them. This also means some earlier files are not perhaps as clean and bug free as they could be. I hope to return to these once the core deliverables for my thesis have been completed.

## A short timeline (updated regularly)
This project was built over several months. 

- February
  1. Initial discussion and familiarization with the project idea
  2. Familiarization with previous code base and work done by former student.
 - March
	 1. Begin work on building a feature store. Basic idea was generate the features necessary in papers [1]-[5]. 
	 2. Began to incorporate testing with [PyTest](https://docs.pytest.org/en/stable/contents.html) 
 - April
	 1. Some major improvements on code quality:
		1.  Started using [Ruff](https://github.com/astral-sh/ruff) for linting the code (linting is exactly as it sounds, like you would lint a jacket for debris, Ruff will lint your code for stuff you don't need and stuff you are missing.)
		2. **I wanted to get better at thinking ahead.**
			 1. After listening to some podcasts ([here](https://youtu.be/I845O57ZSy4?si=J0gQ8SkIAPgIORpR) and [here](https://youtu.be/tNZnLkRBYA8?si=7t4ilfG3zgnjLX2c)) with world-class engineers, decided to use `assert` statements everywhere. At minimum , 2 for each function following the [NASA principles for safety-critical code](https://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code). I aim to adhere  to atleast rules 1, 2, 4, and 5.   
			 2. Started adding type hints to both function calls, and returns as well as any variables that I am declaring anywhere. I want to get used to this, and hopefully soon get familiar with C/C++ which has even stricter requirements (for which the original NASA rules were designed).
		 3. Started using coverage tests with `PyTest -cov`. Essentially, this checks which parts of your code are being hit by your tests and which are not ( more my experience with this here)


## References
1.  Tang, S., Dunnmon, J. A., Saab, K., Zhang, X., Huang, Q., Dubost, F., ... & Lee-Messer, C. (2021). Automated seizure detection and seizure type classification from electroencephalography with a graph neural network and self-supervised pre-training. arXiv preprint arXiv:2104.08336, 10.
    
2.  Li, Y., Chen, J., Li, F., Fu, B., Wu, H., Ji, Y., ... & Zheng, W. (2022). GMSS: Graph-based multi-task self-supervised learning for EEG emotion recognition. IEEE Transactions on Affective Computing, 14(3), 2512-2525.
    
3.  Qiu, L., Zhong, L., Li, J., Feng, W., Zhou, C., & Pan, J. (2024). SFT-SGAT: A semi-supervised fine-tuning self-supervised graph attention network for emotion recognition and consciousness detection. Neural Networks, 180, 106643.
    
4.  Zeng, Y., Lin, J., Li, Z., Xiao, Z., Wang, C., Ge, X., ... & Liu, M. (2024). Adaptive node feature extraction in graph-based neural networks for brain diseases diagnosis using self-supervised learning. NeuroImage, 297, 120750.
    
5.  Van Dijk, H., Van Wingen, G., Denys, D., Olbrich, S., Van Ruth, R., & Arns, M. (2022). The two decades brainclinics research archive for insights in neurophysiology (TDBRAIN) database. Scientific Data, 9(1), 333. https://doi.org/10.1038/s41597-022-01474-2x
