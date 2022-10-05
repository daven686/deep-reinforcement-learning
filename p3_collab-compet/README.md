[//]: # (Image References)


# Project 3: Collaboration and Competition

### Project Details

In the current project, the goal was to train **two** agents which communicate with each other. In this context, an agent corresponds to a tennis racket which can bounce a ball over a net. A reward of $R_t=0.1$ is provided to an agent for hitting the ball over the net, whereas an agent receives a reward of $R_t=-0.01$ for letting the ball hit the ground or hitting the ball out of bounds. Thus, in order to optimize the total score, the aim of the agents is to keep the ball in play for as many time steps as possible. 

The state space $\mathcal{S}$ has $8$ dimensions corresponding to position and velocity of the ball and racket. Based on this information and interaction with the respective other agent, an agent has to learn how to select actions in the best way. 

The action space $\mathcal{A}$ is continuous since two actions are available, corresponding to movement toward/away from the net, and jumping. The task is episodic, i.e. $t\in[0,1,\ldots,T]$.

In order to pass, the agents need an average score of $0.5$ over $100$ consecutive episodes (after taking the maximum over both agents). To be more precise, after each episode $j$, add up the rewards $R_{t}^{(i,j)}$ that each agent $i$ received to get an individual score $$G_{i,j}=\sum_{t=0}^TR_{t}^{(i,j)}$$ for each agent. This yields $2$ scores $G_{1,j},G_{2,j}$. Then, take the maximum $$M_j=\max_i{G_{i,j}}$$ of these $2$ scores leading to a single score $M_j$ for each episode $j$. The environment is considered solved, when the average over $100$ episodes of those average scores $M_j$ is at least $0.5$.
### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p3_collab-compet/` folder, and unzip (or decompress) the file. 

### Instructions

Follow now the instructions in `Tennis.ipynb` to get started with training your own agent!  

What you can also do is to explore the performance of already pre-trained agents. Here, we provide two different network architectures for the underlying Actor-Critic Models.

- Architecture 1: 
  - **Actor_2HL_BN**: a neural net with two hidden layers. The second layer (first hidden) has $400$ nodes, the third layer (second hidden) $300$ nodes. Activation function is ReLU, except for the last one which is $\tanh$. Additionally, we apply batch-normalization to the first hidden layer, see [this paper](https://arxiv.org/pdf/1502.03167.pdf) for further information. 
  - **Critic_2HL_BN**: a neural net with two hidden layers. The second layer (first hidden) has $400$ nodes, the third layer (second hidden) $300$ nodes. For details, we refer to the Python code. Activation function is ReLU, again. Additionally, we apply batch-normalization to the first hidden layer. 

- Architecture 2:
  - **Actor_1HL**: a neural net with one hidden layer. The second layer (first hidden) has $400$ nodes. Activation function is ReLU, except for the last one which is $\tanh$.
  - **Critic_3HL_leaky**: a neural net with three hidden layers. The second layer (first hidden) has $300$ nodes, the third layer (second hidden) $200$ nodes, the fourth layer (third hidden) $100$ nodes and finally the fifth layer $1$ node. For details, we refer to the Python code. Activation function here is **leaky** ReLU.

In order to investigate the models, please use the following files:

- `actor_critic_networks.py`: code for the neural networks which are used as actor models and critic models
- `deep_det_policy_grad_agent.py`: code for single agent; parts of the algorithm Deep Deterministic Policy Gradient (DDPG) 
- `deep_det_policy_grad_multi_agents.py`: code for the agents and their interaction; algorithm is the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
- `Tennis_solution.ipynb`: Jupyter notebook containing the solution
- `actor_2x2_BN_{0/1}.pth`: model weights of pre-trained "Actor_2HL_BN"-model (0=left racket, 1=right racket). 
- `actor_1x3_leaky_{0/1}.pth`: model weights of pre-trained "Actor_1HL"-model (0=left racket, 1=right racket). 
- `critic_2x2_BN_{0/1}.pth`: model weights of pre-trained "Critic_2HL_BN"-model (0=left racket, 1=right racket). 
- `critic_1x3_leaky_{0/1}.pth`: model weights of pre-trained "Critic_3HL_leaky"-model (0=left racket, 1=right racket). 

Please make sure that you fulfill the requirements with respect to the libraries in the `requirements.txt` file of the folder `python/`, too!