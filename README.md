# Introduction to my Udacity Continuous Control Project
This readme attempts to describe what you need to run my solution in ADDITION / SUPPLIMENTAL to the basic Udacity 2nd Project for the Reinforcement Learning class Continuous Control project P2_Continuous-Control readme information.

Briefly the project uses the Unity (MS Visual Studios) pre-defined environment (Reacher.exe) which is double jointed arm which can move to target locations. Goal is to get the agent to maintain its position at the target location for as many time steps as possible, with a reward of +0.1 for each step that the agent's hand is in the goal location.

Observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocityies of the arm. Each action is a vector of 4 numbers, corresponding to torque applied to two joints. Every entry to the action vector should be a number between -1 and 1.

The further details of this project is contained in this directory in the readme1st.md file, which was the Udacity original readme file for this project, which I renamed to avoid conflict/confusion. 

# Project environment details 
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This project is considered complete when the 100 average of the training scores exceed 30.

Two choices were given to execute, one with a single arm and one with 20 arms. We chose the former given that we are a little late on the project.

I've also included the Udacity .gitignore, CODEOWNERS, and LICENCE files for the entire Reinforcement Learning class as well as the class readme file, renamed to readme2nd.md again to avoid conflicts with this file (project) readme file and my provided readme file.

# Brief description of setting up the environment
This development was performed in Windows 64bit environment, so if you have a different computer environment you may need (slightly) different instructions, particularly with regards to the Unity Reacher.zip file and setting the env as the Reacher application (Reacher.exe for Windows).

If you have already set up this environment for the p1_navigation project, you can skip most of the one-time setup of the conda environment, MS Visual Studios, etc below and only focus on installation of the Reacher_Windows_x86_64.zip file, as most of this section only needs to be done once.

Although details of setting up this environment can be found in the Readme1st.md and Readme2nd.md(Dependencies section), briefly it involves:

1) downloading the Reacher_Windows_x86_64.zip file containing the self-contained unity environment 
2) put the resulting directory in the p2_continuous-control folder; we further placed the Reacher.exe file in the p2_continuous-control top folder
3) we also followed the udacity README.md file concerning the setup of the (CONDA) environment for the dqn alone (one-time only):
	a) conda create --name drlnd python=3.6 
	b) activate drlnd
	c) use the drlnd kernel in the Jupyter notebook when running this project
4) We installed MS Visual Studios 2017 & 2022 (one-time only). We did not find the "Build Tools for Visual Studio 2019" on the link provided (https://visualstudio.microsoft.com/downloads/) as indicated in the provided instructions, but rather mostly VS 2022 (VS_Community.exe) and some other things. We selected Python and Unity aspects of the download to hopefully cover our bases there and that seemed to work.
5) Clone the repository locally (if you haven't already; I had) and pip install the python requirements (last line, c)) (one-time only):
	a)git clone https://github.com/udacity/deep-reinforcement-learning.git
	b) cd deep-reinforcement-learning/python
	c) pip install .
6) pip install unityagents
	a) which may require downloading the unity app for personal individual use (one-time only): https://store.unity.com/front-page?check_logged_in=1#plans-individual

We have provided the Reacher.exe, ReacherData  and the python directory within the repository for convenience

# My model description

Briefly my model is strongly based on the Udacity DDPG (Deep Deterministic Policy Gradient) lessons in the actor-critic sections, specifically the pendulum miniproject.

We initially considered amending this implementation to include some of the more effective modifications found in the literature; however, most of the key ones we were interested in weren't required to acheive the required performance. 

We reviewed some of the suggested articles on continous control, especially "Benchmarking Deep Reinforment Learning for Continous Control, Duan et al, 2016, as suggested at the end of the lesson. "Continous control with deep reinforcement learning, Lillicrap et al, 2019, and "Distributed Distributional Deterministic Policy Gradients, Barth-Maron et al, 2018. 

In addition at the end of lessons, the instructor reviewed DeepRL by ShangtonZhang on github and we reviewed his provided examples.py file in some detail for a wide variety of different reinforcement learning techniques; however in the end we did not use this although it was helpful in providing some guidance.

Finally and perhaps most importantly we were running into trouble with having 20 agents (e.g., expected array got a dict) and certain aspects of our agents class, so we searched the internet for "how to handle mutliple agents in ddpg pytorch" and found the following link: 
https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents
by Mike Richardson
with the note at the top of the python files:
""""
Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
"""

We focused on the multi-agent aspects of his solution in the agents class to help us with our error message.

Based on these papers and example tools, we decided to modify the DDPG based on the lessons, the benchmark implementation discussion, and the "Distributed Distributional Deterministic Policy Gradients" paper to include 
1) Single Replay buffer for potentially multiple distributed parallel actors
2) soft updates
3) target networks for both the critic and policy when forming the regression target.

We were also interested in including the following based on the papers we read but in the end they weren't required:
1) N-step sampling 
2) clipped gradient
3) wait c steps before updating the actor and critic networks

The DDPG allows learning to occur with a fixed set of weights and then updating the weights in the other sets, and then once the training is done, it then updates the fixed weights NN and starts again. This tends to avoid overtuning/overfitting.


## PRELIMINARY NOTES:
The provided random actions section of code that was originally provided we commented out (just to show it present as the original), and instead began substitution of the necessary packages and tools in the provided jupyter notebook.

We started with just importing the model and agent from DDPG as before. Locally we've renamed the python files ddpg_model.py and ddpg_agent.py. 


## APPROACH
--> UPDATE: We started with the default (intial) values for the previous DDPG. 

The two included python files are ddpg_model.py, which contains the DDPG implementation (only seen by the agent) from DDPG, and ddpg_agent.py, which contains the Agent class as well as a supporting ReplayBuffer class to store the experiences in tuples for use by the DDPG.

The NN is composed of 3 fully connected layers with two NN internal sizes (defaults: fc1_size=64 and fc2_size=64) using RELU activation functions along with an initial state_size and a final action_size to map into the reachers input (state) and output (action) environment. The DDPG has an __init__ function to be invoked on class creation and the forward method using the NN's to convert the current state into an action.

Here we used the same structure we had for the DDPG for the pendulum problem, but based on the paper comments we reduced the size of the layers to 256 and 128, respectively.


### Agent: the initial agent solution used in the DDPG mini project was used as-is with the following Hyperparamters:
Buffer size: 100,000
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
batch size: 64

This resulted in no apparent gain in performance with average score less than 1. As a result we decreased the critic weighting to match that of the actor and reran

### Agent: the final agent solution used in the DDPG mini project decreased the LR_CRITIC by a factor of 10 
Buffer size: 100,000
BATCH_SIZE = 128        # minibatch size
#BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
#LR_CRITIC = 1e-3        # learning rate of the critic
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
batch size: 64

The Agent class itself is composed of an __init__ fuction for construction, which creates the two actor-critic networks, one that is local and one that is the target network, along with the optimizer and memory buffer from the ReplayBuffer class to store experiences and a noise source (Orstein-Uhlenbeck, from the original DDPG mini-project) class as well. 

The Agent step method adds the current experience into the memory buffer for each agent, and stores the experience into memory and exectutes the learn fucntion once there are enough experiences.

The Agent act method returns actions for a given state given current policy. When it is not learning (no_grad()) it retrieves the actions for each actor from the local (actor) network and the current states. It does this by evaluating (eval) the local actor network, get new actions from the local network, train the local network if applicable, and finally select actions with noise.

The Agent learn method was the one for which we had to provide the appropriate solutions previously with the DDPG mini-project and is essentially unchanged. Here we unpack the tuple experiences into states, actions, rewards, next_states, and dones. The next_states are used in the target (NOT local) qnetwork to get the next target actions. These are then detached from the resulting tensor to make a true copy, access Qtable for the next action, and hence the rewards of the target network. We then get the next action results from the local network and then determine the MSE loss between the target and local network fits. We then zero_grad the optimizer, propagate the loss backwards through the network, and perform a step in the optimizer. Finally a soft update is performed on the target network, using TAU times the local network parameters and (1-TAU) times the target network parameters to update the target network parameters.

As indicated the original DDPG, the agent has a helper class ReplayBuffer, with methods add, to add experiences to the buffer, and sample, to sample experiences from the buffer, and is used extensively in the step method for the Agent class.

Originally we expected to look at some of the post-DDPG example approaches, especially the N-step, clipping and, based on the the background lesson, to wait for C epsisodes before updating in the critic network. However since these were mainly modifications of the internal workings of the agents and the like, we felt that it was best to first get the baseline DDPG running and then see if there are problems about possibly making these modifications. 

So we start with our original agent and model, which we've imported locally and import the (slightly modified) DDPG function for the unity setup, and this was found to be sufficient for this exercise.

# Running the model
To run the code, download my entire repository onto your local drive, which includes the Reacher.zip file that you'll want to unpack locally, and copy the Reacher.exe into the project top folder for the self-contained unity environment. You will probably want to make sure you have a recent version of MS Visual Studios (2017 to 2022 seemed to be OK) and use your Anaconda powershell to create the drlnd anaconda environment, if you haven't already from the first project. In Anaconda,  click on the "Applications on " pull-down menu and select your newly created drlnd environment (drlnd) and once that loads then launch the Jupyter notebook from that particular environment. 

Once in the notebook, you'll want to go the the kernel and at the bottom change your kernel from python to your newly created drlnd. At this point you are ready to run the notebook.

At this point I usually select restart and run all to make sure all the cells will run without interruptions.

A few additional notes that caused some confusion and delay on my part
1) I found it safe / best to always restart and clear output EACH time you try to run. I often got weird errors that didn't make any sense that went away as soon as I cleared output
2) You have to click on the Unity window that opens up (or possibly minimize it); otherwise you'll get a frustrating timeout that doesn't make any sense.


We made our requirement at episode 109 with learning score of 30.21:
Requirement met on Episode 109	Requirement Average Score: 30.21
Ran the episodes to exceed 33 to episodes 118:
Final Episode 118	Final Average Score: 33.23

and when re-ran the weights for a single run got a score of 38.9:
current mean score: 38.91899913009256

## FUTURE IMPROVEMENTS
As mentioned above we believe probably the greatest improvement will come from an N-step implementation:
1) N-step sampling 
Since this could provide information on longer-term solutions earlier

Another would be to do at least a sensitivity exploration of the hyperparameters. Of particular interest are the following in order of interest:
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
TAU = 1e-3              # for soft update of target parameters
GAMMA = 0.99            # discount factor
For example, changing the LR_CRITIC by an order of magnitude changed the result from < 1 after 175 episodes to > 30 in just over 100 episodes. Learning rates we suspect could further reduced and refined. Tau also was not explored and could also be adjusted (higher at lower iterations and lower at higher iterations) and finally the discount factor could be increased/decreased to see if there is a better value

Another possible impact would be to hold for a certain number steps (C) before updating
3) wait c steps before updating the actor and critic networks

We don't think clipping the gradient appears to have much impact as we don't seem to be falling off a cliff while learning, but this would be another area to look
2) clipped gradient



# REFERENCES
##### Basic
"Deep Reinforcement Learning: Pong from Pixels"
http://karpathy.github.io/2016/05/31/rl/

Github example of continous control with DDPG
https://github.com/MariannaJan/continous_control_DDPG

Article: "Continuous Deep Q-Learning with Model-based Acceleration", by Gu et al., 2016
https://arxiv.org/pdf/1603.00748.pdf

Article: "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING" by Lillencrap et al, 2019
https://arxiv.org/pdf/1509.02971.pdf

Article: "Benchmarking Deep Reinforcement Learning for Continuous Control" by Duan et al, 2016
https://arxiv.org/pdf/1604.06778.pdf


"Let’s make a DQN: Double Learning and Prioritized Experience Replay", 2016
https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/


Github RL Lab by by ShangtonZhang
https://github.com/rll/rllab

"DDPG (Actor-Critic) Reinforcement Learning using PyTorch and Unity ML-Agents", solution to 
"how to handle mutliple agents in ddpg pytorch" google search
https://github.com/xkiwilabs/DDPG-using-PyTorch-and-ML-Agents



##### D4PG
Article: "DISTRIBUTED DISTRIBUTIONAL DETERMINISTIC POLICY GRADIENTS",by Barth-Maron, 2018
https://openreview.net/pdf?id=SyZipzbCb
