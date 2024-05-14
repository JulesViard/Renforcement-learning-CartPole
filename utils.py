import random
from collections import deque

import torch
import torch.nn.functional as F
from gym.core import Env
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
    
    Returns:
        Float scalar tensor with loss value
    """

    bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
    q_values = policy_dqn(states).gather(1, actions).reshape(-1)
    return ((q_values - bellman_targets)**2).mean()


# Epsilon decay function:
def eps_decay(index_ep, steep):
    """
    Calculate the exponential decay of epsilon based on the episode index.

    This function computes the epsilon value for the epsilon-greedy. 

    Parameters:
    index_ep (int): The current episode index.
    steep (float): The steepness parameter for the decay. A smaller value results in a faster decay of epsilon.

    Returns:
    float: The decayed epsilon value for the given episode index.
    """
    eps = np.exp(-(index_ep/steep))
    return eps

def plot_eps_decay(steep, episode_nb):
    """
    Plot the curve of epsilon values over a number of episodes using exponential decay.

    Parameters:
    steep (float): The steepness parameter for the epsilon decay function (eps_decay).
    episode_nb (int): The total number of episodes for which the epsilon values are plotted.

    Returns:
    None: This function does not return a value but displays a plot.
    """
    test_eps_decay = []
    for i in range(episode_nb):
        test_eps_decay.append(eps_decay(i,steep))

    plt.plot(test_eps_decay)
    plt.xlabel("episode index")
    plt.ylabel("epsilon value")
    plt.title("Epsilon curve")
    plt.show()



# Plot Experimentation:
def plot_results(runs_results, parameters):
    """
    Plot the learning curve across episodes, by calculating the average reward and the standard deviation.

    Parameters:
    runs_results (list of lists): A nested list where each inner list contains the durations or returns for each episode in a single run.
    parameters (dict): A dictionary containing parameters used in the experiments. Key parameters are displayed in the plot as an annotation.

    Returns:
    None: This function does not return a value but displays a plot.

    Note:
    - Hyperparameters are displayed in a text box.
    - The function marks a horizontal line at a return value of 100, representing the performance threshold.
    """

    fig, ax = plt.subplots()

    results = torch.tensor(runs_results)
    means = results.float().mean(0)
    stds = results.float().std(0)

    nb_episode = parameters["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means)
    plt.ylabel("return")
    plt.xlabel("episode")
    plt.fill_between(np.arange(nb_episode), means, means + stds, alpha=0.3, color='b')
    plt.fill_between(np.arange(nb_episode), means, means - stds, alpha=0.3, color='b')

    # Hyperparameter text box:
    textstr = '\n'.join((
        f"NN = {parameters['structure_DQN_policy']}",
        f"size_buffer = {parameters['size_buffer']:.2f}",
        f"size_batch = {parameters['size_batch']:.2f}",
        f"nb_episodes = {nb_episode:.2f}",
        f"freq_update_target = {parameters['freq_update_target']:.2f}",
        f"epsilon = {parameters['epsilon']:.2f}",
        f"learning_rate_start = {parameters['learning_rate_start']:.2f}",
        f"step_size_lr = {parameters['step_size_lr']:.2f}",
        f"gamma_lr = {parameters['gamma_lr']:.2f}",
        f"wdecay = {parameters['w_dcy']:.4f}"
    ))

    plt.axhline(y=100, color='r', linestyle='-')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(1.05, 0.9, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    plt.show()



# Plot for buffer and batch:
def plot_comparison_buffer_batch(run1, parameters1, run2, parameters2, run3, parameters3):
    """
    Plot a comparison of the performance of three sets of runs with different batch sizes for a given buffer size.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters used in the first set of runs.
    run2 : Similar to run1, but for the second set of runs.
    parameters2 (dict): Parameters for the second set of runs.
    run3 : Similar to run1, but for the third set of runs.
    parameters3 (dict): Parameters for the third set of runs.

    Returns:
    None: This function does not return a value but displays a comparative plot.

    Note:
    - The plot displays mean returns with lines and their standard deviations with shaded areas.
    - Each set of runs is distinguished by color and is labeled with its corresponding batch size.
    """
    fig, ax = plt.subplots()

    buffer_size_current = parameters1['size_buffer']
    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)
    batch_size_current1 = parameters1['size_batch']

    results2 = torch.tensor(run2)
    means2 = results2.float().mean(0)
    stds2 = results2.float().std(0)
    batch_size_current2 = parameters2['size_batch']

    results3 = torch.tensor(run3)
    means3 = results3.float().mean(0)
    stds3 = results3.float().std(0)
    batch_size_current3 = parameters3['size_batch']

    
    nb_episode = parameters1["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means1, label=f'Batch size = {batch_size_current1}', color='navy')
    plt.fill_between(np.arange(nb_episode), means1, means1 + stds1, alpha=0.3, color='#add8e6')
    plt.fill_between(np.arange(nb_episode), means1, means1 - stds1, alpha=0.3, color='#add8e6')

    plt.plot(torch.arange(nb_episode), means2, label=f'Batch size = {batch_size_current2}', color='forestgreen')
    plt.fill_between(np.arange(nb_episode), means2, means2 + stds2, alpha=0.3, color='#90ee90')
    plt.fill_between(np.arange(nb_episode), means2, means2 - stds2, alpha=0.3, color='#90ee90')


    plt.plot(torch.arange(nb_episode), means3, label=f'Batch size = {batch_size_current3}', color='darkred')
    plt.fill_between(np.arange(nb_episode), means3, means3 + stds3, alpha=0.3, color='#ffcccb')
    plt.fill_between(np.arange(nb_episode), means3, means3 - stds3, alpha=0.3, color='#ffcccb')


    plt.ylabel("return")
    plt.xlabel("episode")
    plt.title(f'Comparison of batch size for buffer size of {buffer_size_current} over ten training runs')
    plt.legend()

    explanation_text = "Line: Mean Average\nShaded Area: ± Standard Deviation"
    fig.text(0.93, 0.5, explanation_text, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))


    plt.axhline(y=100, color='m', linestyle='--', linewidth=0.8)
    plt.show()



# Epsilon decay comparison:
def plot_comparison_eps(run1, parameters1, run2, parameters2, run3, parameters3):
    """
    Plot a comparison of the performance of three sets of runs with different epsilon values.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters used in the first set of runs.
    run2 : Similar to run1, but for the second set of runs.
    parameters2 (dict): Parameters for the second set of runs.
    run3 : Similar to run1, but for the third set of runs.
    parameters3 (dict): Parameters for the third set of runs.

    Returns:
    None: This function does not return a value but displays a comparative plot.

    Note:
    - The plot displays mean returns with lines and their standard deviations with shaded areas.
    - Each set of runs is distinguished by color and is labeled with its corresponding epsilon value.
    """
    fig, ax = plt.subplots()

    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)
    eps1 = parameters1['steep_eps_decay']

    results2 = torch.tensor(run2)
    means2 = results2.float().mean(0)
    stds2 = results2.float().std(0)
    eps2 = parameters2['steep_eps_decay']

    results3 = torch.tensor(run3)
    means3 = results3.float().mean(0)
    stds3 = results3.float().std(0)
    eps3 = parameters3['steep_eps_decay']

    
    nb_episode = parameters1["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means1, label=f'Epsilon = exp(-i/{eps1})', color='navy')
    plt.fill_between(np.arange(nb_episode), means1, means1 + stds1, alpha=0.3, color='#add8e6')
    plt.fill_between(np.arange(nb_episode), means1, means1 - stds1, alpha=0.3, color='#add8e6')

    plt.plot(torch.arange(nb_episode), means2, label=f'Epsilon = exp(-i/{eps2})', color='forestgreen')
    plt.fill_between(np.arange(nb_episode), means2, means2 + stds2, alpha=0.3, color='#90ee90')
    plt.fill_between(np.arange(nb_episode), means2, means2 - stds2, alpha=0.3, color='#90ee90')


    plt.plot(torch.arange(nb_episode), means3, label=f'Epsilon = exp(-i/{eps3})', color='darkred')
    plt.fill_between(np.arange(nb_episode), means3, means3 + stds3, alpha=0.3, color='#ffcccb')
    plt.fill_between(np.arange(nb_episode), means3, means3 - stds3, alpha=0.3, color='#ffcccb')


    plt.ylabel("return")
    plt.xlabel("episode")
    plt.title('Comparison of Exponential decaying over ten training runs')
    plt.legend()

    explanation_text = "Line: Mean Average\nShaded Area: ± Standard Deviation"
    fig.text(0.93, 0.5, explanation_text, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))


    plt.axhline(y=100, color='m', linestyle='--', linewidth=0.8)
    plt.show()




# Learning rate comparison:
def plot_comparison_lr(run1, parameters1, run2, parameters2, run3, parameters3):
    """
    Plot a comparison of the performance of three sets of runs with different learning rate values.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters used in the first set of runs.
    run2 : Similar to run1, but for the second set of runs.
    parameters2 (dict): Parameters for the second set of runs.
    run3 : Similar to run1, but for the third set of runs.
    parameters3 (dict): Parameters for the third set of runs.

    Returns:
    None: This function does not return a value but displays a comparative plot.

    Note:
    - The plot displays mean returns with lines and their standard deviations with shaded areas.
    - Each set of runs is distinguished by color and is labeled with its corresponding learning rate value.
    """
    fig, ax = plt.subplots()

    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)
    lr_current1 = parameters1['learning_rate_start']

    results2 = torch.tensor(run2)
    means2 = results2.float().mean(0)
    stds2 = results2.float().std(0)
    lr_current2 = parameters2['learning_rate_start']

    results3 = torch.tensor(run3)
    means3 = results3.float().mean(0)
    stds3 = results3.float().std(0)
    lr_current3 = parameters3['learning_rate_start']

    
    nb_episode = parameters1["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means1, label=f'Learning rate start = {lr_current1}', color='navy')
    plt.fill_between(np.arange(nb_episode), means1, means1 + stds1, alpha=0.3, color='#add8e6')
    plt.fill_between(np.arange(nb_episode), means1, means1 - stds1, alpha=0.3, color='#add8e6')

    plt.plot(torch.arange(nb_episode), means2, label=f'Learning rate start = {lr_current2}', color='forestgreen')
    plt.fill_between(np.arange(nb_episode), means2, means2 + stds2, alpha=0.3, color='#90ee90')
    plt.fill_between(np.arange(nb_episode), means2, means2 - stds2, alpha=0.3, color='#90ee90')


    plt.plot(torch.arange(nb_episode), means3, label=f'Learning rate start = {lr_current3}', color='darkred')
    plt.fill_between(np.arange(nb_episode), means3, means3 + stds3, alpha=0.3, color='#ffcccb')
    plt.fill_between(np.arange(nb_episode), means3, means3 - stds3, alpha=0.3, color='#ffcccb')


    plt.ylabel("return")
    plt.xlabel("episode")
    plt.title('Comparison of Learning Rate impact over ten training runs')
    plt.legend(loc='upper left')

    explanation_text = "Line: Mean Average\nShaded Area: ± Standard Deviation"
    fig.text(0.93, 0.5, explanation_text, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.axhline(y=100, color='m', linestyle='--', linewidth=0.8)
    plt.show()


def plot_learning_evolution(parameters1):
    """
    Plot the evolution of the learning rate over a series of episodes.

    Parameters:
    parameters1 (dict): A dictionary containing hyperparameters

    Returns:
    None: This function does not return a value but displays a plot.
    """
    start_lr = parameters1["learning_rate_start"]
    list_lr = []
    current_lr = 0
    d = 0

    for i in range(0, parameters1["nb_episode"]):
        if i % parameters1["step_size_lr"] == 0 and i> 0:
             d = d+1
        current_lr = start_lr * (parameters1["gamma_lr"] ** d)
        list_lr.append(current_lr)

    plt.plot(list_lr)
    plt.xlabel("episode index")
    plt.ylabel("learning rate value")
    plt.title("Learning rate curve")
    plt.show()



# Plot weight decay comparison:
def plot_comparison_weightdecay(run1, parameters1, run2, parameters2, run3, parameters3):
    """
    Plot a comparison of the performance of three sets of runs with different weight decay (of AdamW optimiser) values.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters used in the first set of runs.
    run2 : Similar to run1, but for the second set of runs.
    parameters2 (dict): Parameters for the second set of runs.
    run3 : Similar to run1, but for the third set of runs.
    parameters3 (dict): Parameters for the third set of runs.

    Returns:
    None: This function does not return a value but displays a comparative plot.

    Note:
    - The plot displays mean returns with lines and their standard deviations with shaded areas.
    - Each set of runs is distinguished by color and is labeled with its corresponding weight decay value.
    """
    fig, ax = plt.subplots()

    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)
    weight_decay_current1 = parameters1['w_dcy']

    results2 = torch.tensor(run2)
    means2 = results2.float().mean(0)
    stds2 = results2.float().std(0)
    weight_decay_current2 = parameters2['w_dcy']

    results3 = torch.tensor(run3)
    means3 = results3.float().mean(0)
    stds3 = results3.float().std(0)
    weight_decay_current3 = parameters3['w_dcy']

    
    nb_episode = parameters1["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means1, label=f'Weight decay = {weight_decay_current1}', color='navy')
    plt.fill_between(np.arange(nb_episode), means1, means1 + stds1, alpha=0.3, color='#add8e6')
    plt.fill_between(np.arange(nb_episode), means1, means1 - stds1, alpha=0.3, color='#add8e6')

    plt.plot(torch.arange(nb_episode), means2, label=f'Weight decay = {weight_decay_current2}', color='forestgreen')
    plt.fill_between(np.arange(nb_episode), means2, means2 + stds2, alpha=0.3, color='#90ee90')
    plt.fill_between(np.arange(nb_episode), means2, means2 - stds2, alpha=0.3, color='#90ee90')


    plt.plot(torch.arange(nb_episode), means3, label=f'Weight decay adaptative, starting at = {weight_decay_current3}', color='darkred')
    plt.fill_between(np.arange(nb_episode), means3, means3 + stds3, alpha=0.3, color='#ffcccb')
    plt.fill_between(np.arange(nb_episode), means3, means3 - stds3, alpha=0.3, color='#ffcccb')


    plt.ylabel("return")
    plt.xlabel("episode")
    plt.title('Comparison of Weight Decay impact over ten training runs')
    plt.legend(loc='upper left')

    explanation_text = "Line: Mean Average\nShaded Area: ± Standard Deviation"
    fig.text(0.93, 0.5, explanation_text, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1'))

    plt.axhline(y=100, color='m', linestyle='--', linewidth=0.8)
    plt.show()


def plot_weightdecay_evolve(run1, parameters1):
    """
    Plot the evolution with a specific weight decay evolving setting.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters.

    Returns:
    None: This function does not return a value but displays a plot.
    """
    fig, ax = plt.subplots()

    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)
    weight_decay_current1 = parameters1['w_dcy']
    
    nb_episode = parameters1["nb_episode"]
    
    plt.plot(torch.arange(nb_episode), means1, label=f'Weight decay start = {weight_decay_current1}', color='navy')
    plt.fill_between(np.arange(nb_episode), means1, means1 + stds1, alpha=0.2, color='#add8e6')
    plt.fill_between(np.arange(nb_episode), means1, means1 - stds1, alpha=0.2, color='#add8e6')


    plt.ylabel("return")
    plt.xlabel("episode")
    plt.title('Adaptative Weight Decay')
    plt.legend()

    plt.axhline(y=100, color='r', linestyle='-')
    plt.show()



# Plot learning curve:
def plot_result_learning_curve(run1, parameters1):
    """
    Plot the learning curve of over a number of training runs.

    Parameters:
    run1 : List containing the reward for each episode for every training runs previously computed.
    parameters1 (dict): A dictionary containing hyperparameters.
    
    Returns:
    None: This function does not return a value but displays a learning curve plot.

    Note:
    - A horizontal line at a return value of 100, representing the performance threshold.
    """
    fig, ax = plt.subplots()

    results1 = torch.tensor(run1)
    means1 = results1.float().mean(0)
    stds1 = results1.float().std(0)

    nb_episode = parameters1["nb_episode"]

    plt.plot(torch.arange(nb_episode), means1)
    plt.ylabel("return")
    plt.xlabel("episode")
    plt.fill_between(np.arange(nb_episode), means1, means1+stds1, alpha=0.3, color='b')
    plt.fill_between(np.arange(nb_episode), means1, means1-stds1, alpha=0.3, color='b')
    plt.title("Learning Curve for Optimised agent")

    plt.axhline(y=100, color='r', linestyle='--', linewidth=0.5)
    plt.show()


