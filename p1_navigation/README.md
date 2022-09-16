[//]: # (Image References)

[random]: pictures/random.gif "Random Agent"
[trained]: pictures/trained.gif "Trained Agent"

# Project 1: Navigation

### Project Details

In this project, you will train an agent navigating in a two-dimensional box and collect bananas. To be more precise, the agent can meet with two different types of bananas, yellow and blue.   

A reward of $R_t=1$ is provided for getting a yellow banana, and a reward $R_t=-1$ is provided for meeting a blue banana. Thus, in order to optimize the total score, the aim of your agent is to collect as many yellow bananas as possible while avoiding blue bananas. In the following, you can see the difference between random (left) and a trained (right) behaviour of an agent:  

![Random Agent][random] ![Trained Agent][trained]

The state space $\mathcal{S}$ has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Based on this information, the agent has to learn how to select actions in the best way.  Four discrete movements can be made, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

Thus, the action space is given by $\mathcal{A}=\{0,1,2,3\}$. The task is episodic, i.e. $t\in[0,1,\ldots,T]$. In order to solve the environment, your agent must get an average score of $+13$ over $100$ consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in your cloned GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow now the instructions in `Navigation.ipynb` to get started with training your own agent!  

What you can also do is to explore the performance of an already pre-trained agent. Here, we provide two different network architectures for the underlying Q-network which estimates the action-value function. 

- **QNetwork_2Hidden**: a neural net with two hidden layers. The first layer has $|\mathcal{S}|=37$ inputs and $64$ outputs, the second layer (first hidden) $64$ inputs and $64$ outputs, the third layer (second hidden) again $64$ inputs and $64$ outputs and finally the fourth layer $64$ inputs and $|\mathcal{A}|=4$ outputs 
- **QNetwork_3Hidden**: a neural net with three hidden layers. The first layer has $|\mathcal{S}|=37$ inputs and $128$ outputs, the second layer (first hidden) $128$ inputs and $64$ outputs, the third layer (second hidden) $64$ inputs and $64$ outputs, the fourth layer (third hidden) again $64$ inputs and $64$ outputs and finally the fifth layer $64$ inputs and $|\mathcal{A}|=4$ outputs 

In order to investigate the models, please use the following files:

- `actionvalue_estimators.py`: code for the neural networks which are used as estimators for the action-value function
- `deepqnet_agent.py`: code for the agent; algorithm is deep Q-network with experience replay and fixed Q-targets
- `Navigation_solution.ipynb`: Jupyter notebook containing the solution
- `agent_via_2layers.pth`: model weights of pre-trained "QNetwork_2Hidden"-model. Adapt the file path in `Navigation_solution.ipynb` accordingly and put `agent = agent_via_2layers` before running the episode 
- `agent_via_3layers.pth`: model weights of pre-trained "QNetwork_3Hidden"-model. Adapt the file path in `Navigation_solution.ipynb` accordingly and put `agent = agent_via_3layers` before running the episode 

Please make sure that you fulfill the requirements with respect to the libraries in the `requirements.txt` file of the folder `python/`, too!
