[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# DQN/DDQN Reinforcement Learning Agent for Navigation

### Introduction

This project is taken from the [Udacity DRLND nano degree program](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) and implements an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]
([source](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif))

#### General setting (see also the original [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))

The agent navigates in a predefined [*Unity* environment](https://github.com/Unity-Technologies/ml-agents) that represents a square world were it has to collect bananas. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space of the agent's environment is represented by a 37 dimensional vector or real numbers and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the environment is considered solved if the agent is able to achieve an average score of +13 over 100 consecutive episodes.

#### Deep reinforcement learning agent

This repository provides an agent implementation that can be trained to solve the task described in [General settings]($general-settings) using deep reinforcement learning. In particular implementations of two variants of neural network based Q-learning algorithms (see e.g. [Q-learning](https://en.wikipedia.org/wiki/Q-learning)) are provided:
- Q-learning with a DNN as approximator (DQN) for the Q-function using experience replay,
- Double Q-learning with a DNN as approximator (DDQN)  using experience replay.

In addition to this algorithm variants two different architectures of DNNs as function approximators are implemented,
- a simple variant with two hidden layers of 64 units each, and
- a more advanced autoencoder-like architecture that is extended with two additional hidden layers with 16 and 8 units at the  input side of the DNN.

The implementation can execute learning and compare a selection of combinations of algorithm variant and network architecture. Once the agent is trained it can replay an episode of using the parameters stored after learning. A discussion of the results can be found in `Report.ipynb` (or `Report.html` respectively). For a description on how to run the program see the [Instructions](#instructions) section below.

### Getting started (see also the original [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation))

To set up your python environment to run the code in this repository, follow this steps:

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__:
	```bash
	conda create --name banananav python=3.6
	source activate banananav
	```
	- __Windows__:
	```bash
	conda create --name banananav python=3.6
	activate banananav
	```

3. Then install the dependencies from the requirements.txt file in this repository:
```bash
pip install .
```

4. Download the *Unity* environment from one of the links below. You must select **only** the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

5. Place the file in the `banananav/resources/` folder, and unzip (or decompress) the file.

6. To be able to run the `Report.ipynb` notebook, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `banananav` environment.  
```bash
python -m ipykernel install --user --name banananav --display-name "banananav"
```

7. Before running code in the notebook, change the kernel to match the `banananav` environment by using the drop-down `Kernel` menu.

### Instructions

The main script in the repository is `banananavigation.py`, which can be invoked to train the agent and store the result, as well as to replay the agent on a single episode from the stored learning results.

After all dependencies are set up as described in the [Getting started](#getting-started) section, the script can be invoked from the root directory of the repository as e.g.
```bash
> python banananavigation.py --settings DQN,DDQN,DQN_SIMPLE,DDQN_SIMPLE --episodes 1000
```
to execute learning and
```bash
> python banananavigation.py --replay
```
When invoked without command line parameters, the script will execute learning for all algorithm + network architecture combinations on 2500 episodes each. For help on the possible command line options execute the script with the --help option.

### Results

Results of training the different algorithm and models can be found in the *results/* directory. In particular the *results/xxx_model.parameters* files contain the trained parameters of the different models.

The included results achieve the target of an average score of +13 in less than 600 episodes and reach a top average score of +17.4. The parameters for the model with the best result are also stored in the *ddqn_model.parameters.pt* file in the root of the repository.
