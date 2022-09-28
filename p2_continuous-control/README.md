[//]: # (Image References)

[random]: pictures/random.gif "Random Agent"
[trained]: pictures/trained.gif "Trained Agent"

# Project 2: Continuous Control

### Project Details

In this project, you will train one **or** several agents in parallel where an agent corresponds to a double-jointed arm which can move to target locations. A reward of $R_t=0.1$ is provided to an agent for each step staying with its hand within its green target location. Thus, in order to optimize the total score, the aim of your agent(s) is to stay in its/their position(s) at the target location for as many time steps as possible. In the following, you can see the difference between random (left) and trained (right) behaviour of $20$ agents:  

![Random Agent][random] ![Trained Agent][trained]

The state space $\mathcal{S}$ has $33$ dimensions and contains the agent's velocity, rotation, position as well as angular velocities of the arm. Based on this information, the agent has to learn how to select actions in the best way. Each action is a vector with four numbers, corresponding to torque applicable to the two joints. Every entry in the action vector is a number between $-1$ and $1$. Therefore, the action space is given by $\mathcal{A}=[-1,1]\times[-1,1]\times[-1,1]\times[-1,1]$ and is thus continuous. The task is episodic, i.e. $t\in[0,1,\ldots,T]$. 

For this project, you are provided with two separate versions of the Unity environment:
- The first version contains a single agent. In order to pass, the agent needs an average score of $30$ over $100$ consecutive episodes.
- The second version contains $20$ identical agents, each with its own copy of the environment. In order to pass, the agents need an average score of $30$ over $100$ consecutive episodes, and over all $20$ agents. To be more precise, after each episode, add up the rewards that each agent received to get a score for each agent. This yields $20$ scores. Then, take the mean of these $20$ scores leading to an average score for each episode. The environment is considered solved, when the average over $100$ episodes of those average scores is at least $30$. 

**Of course, as already mentioned above, you only have to solve one of the two versions of the environment to pass.**

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

Follow now the instructions in `Continuous_Control.ipynb` to get started with training your own agent!  

What you can also do is to explore the performance of already pre-trained agents (version 2). Here, we provide two different network architectures for the underlying Actor-Critic Model.

- Architecture 1: 
  - **Actor_2HL_BN**: a neural net with two hidden layers. The first layer has $|\mathcal{S}|=33$ nodes, the second layer (first hidden) $128$ nodes, the third layer (second hidden) $256$ nodes and finally the fourth layer $|\mathcal{A}|=4$ nodes. Activation function is ReLU, except for the last one which is $\tanh$. Additionally, we apply batch-normalization to the first hidden layer, see [this paper](https://arxiv.org/pdf/1502.03167.pdf) for further information. 
  - **Critic_2HL_BN**: a neural net with two hidden layers. The first layer has $|\mathcal{S}|=33$ nodes, the second layer (first hidden) $128$ nodes, the third layer (second hidden) $256$ nodes and finally the fourth layer $1$ node. For details, we refer to the Python code. Activation function is ReLU, again. Additionally, we apply batch-normalization to both hidden layers. 

- Architecture 2:
  - **Actor_1HL**: a neural net with one hidden layer. The first layer has $|\mathcal{S}|=33$ nodes, the second layer (first hidden) $256$ nodes and finally the third layer $|\mathcal{A}|=4$ nodes. Activation function is ReLU, except for the last one which is $\tanh$.
  - **Critic_3HL_leaky**: a neural net with three hidden layers. The first layer has $|\mathcal{S}|=33$ nodes, the second layer (first hidden) $256$ nodes, the third layer (second hidden) $256$ nodes, the fourth layer (third hidden) $128$ nodes and finally the fifth layer $1$ node. For details, we refer to the Python code. Activation function here is **leaky** ReLU.

In order to investigate the models, please use the following files:

- `actor_critic_networks.py`: code for the neural networks which are used as actor models and critic models
- `deep_det_policy_grad_agent.py`: code for the agent; algorithm is the Deep Deterministic Policy Gradient (DDPG) 
- `Continuous_Control_solution.ipynb`: Jupyter notebook containing the solution
- `actor_2x2_BN.pth`: model weights of pre-trained "Actor_2HL_BN"-model. Adapt the file path in `Continuous_Control_solution.ipynb` accordingly and put `agent = actor_critic_2x2_BN` before running the episode 
- `actor_1x3_leaky.pth`: model weights of pre-trained "Actor_1HL"-model. Adapt the file path in `Continuous_Control_solution.ipynb` accordingly and put `agent = actor_critic_2x2_BN` before running the episode 
- `critic_2x2_BN.pth`: model weights of pre-trained "Critic_2HL_BN"-model. Adapt the file path in `Continuous_Control_solution.ipynb` accordingly and put `agent = actor_critic_1x3_leaky` before running the episode 
- `critic_1x3_leaky.pth`: model weights of pre-trained "Critic_3HL_leaky"-model. Adapt the file path in `Continuous_Control_solution.ipynb` accordingly and put `agent = actor_critic_1x3_leaky` before running the episode 

Please make sure that you fulfill the requirements with respect to the libraries in the `requirements.txt` file of the folder `python/`, too!