import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """
    Approximation of the value function V of a state given as input
    FC network with 1 hidden layers and ReLU activations
    Class used as 'critic'
    Inputs : 
    input_size : state space dimension
    hidden_size : number of hidden neurons
    output_size : 1 (dimension of the value function estimate)

    Custom policy model network for discrete action space
    Inputs : 
    input_size : state space dimension
    hidden_size : nb of hidden neurons 
    action_size : action space dimension

    Policy model network for continuous action space
    Inputs : 
    input_size : state space dimension
    hidden_size : nb of hidden neurons
    action_size : action space dimension
    """

    def __init__(self, input_size, hidden_size_actor, hidden_size_critic, action_size):
        super(ActorCritic, self).__init__()

        # critic network
        self.value_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size_critic),
                nn.ReLU(),
                nn.Linear(hidden_size_critic, hidden_size_critic),
                nn.ReLU(),
                nn.Linear(hidden_size_critic, 1)
                )

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size_actor),
                nn.Tanh(),
                nn.Linear(hidden_size_actor, hidden_size_actor),
                nn.Tanh(),
                nn.Linear(hidden_size_actor, action_size),
                nn.Softmax(dim=-1)
                )

    def forward(self, x):
        raise NotImplementedError

    def select_action(self, x, memory=None,store=True):
        action_probs = self.action_layer(x)
        dist = Categorical(action_probs)
        action = dist.sample()

        if store:
            assert memory is not None
            memory.actions.append(action) 
            memory.logprobs.append(dist.log_prob(action))
    
        return action.item()

    
    def predict(self, x):
        return self.value_layer(x).cpu().detach().numpy()[0]

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.value_layer(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


'''
class ActorContinuous(nn.Module):


    def __init__(self, input_size, hidden_size, action_size, std, env):
        super(ContinuousActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.std = std
        self.env = env
        
    def forward(self, x):
        out = torch.tanh(self.fc1(x.float()))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out
    
    def select_action(self, x): 
        sampled_a = torch.normal(self(x), self.std).detach().numpy()
        sampled_a = max(self.env.action_space.low, sampled_a)
        sampled_a = min(self.env.action_space.high, sampled_a)
        return sampled_a
'''