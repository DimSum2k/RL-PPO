import gym
import Box2D
from gym.spaces import Discrete

import numpy as np 
import torch
from torch import optim
import torch.nn as nn

from networks import ActorCritic


class PPO(object):
    def __init__(self, config, env, device):
            
        self.config = config
        self.device = device
        self.epochs = config["epochs"]
        
        self.gamma = config['gamma'] 
        self.lambd = config['lambda'] 
        self.c1 = config['c1'] 
        self.c2 = config['c2'] 
        self.eps_clipping = config['eps_clipping']
        
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']

        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        env.seed(config['seed'])
        
        # actor critic  (get a second network ? )
        self.actorcritic = ActorCritic(env.observation_space.shape[0], 
                          config['actor']['hidden'],
                          config['critic']['hidden'], 
                          env.action_space.n).to(self.device)
    
        self.optimizer = optim.Adam(self.actorcritic.parameters(), lr=config['actor']['lr'])

        self.MseLoss = nn.MSELoss()
        
    def update(self, memory, obs):
        
            # GAE
            obs_t = torch.from_numpy(obs).float().to(self.device) # s_t+1
            next_value = self.actorcritic.predict(obs_t) # v(s_t+1)
            returns_, advantages_ = self._build_GAE(memory, next_value) # normalize ? 
            loss_val, dry_loss_val, entrop_val = self.optimize_model(memory, returns_, advantages_)
                
            return loss_val, dry_loss_val, entrop_val


        
    def _build_GAE(self, memory, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network
        
        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """
        
        returns, advantages = [], []
        gae = 0 
        
        for i in reversed(range(len(memory))):
            delta = memory.rewards[i] + self.gamma*next_value*(1-memory.dones[i]) - memory.values[i]
            gae = delta + self.gamma*self.lambd*(1-memory.dones[i])*gae
            returns.insert(0,gae + memory.values[i])
            advantages.insert(0,gae)
            next_value = memory.values[i]

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - torch.FloatTensor(memory.values).to(self.device)
            
        return returns, advantages 
    
    
    def optimize_model(self, memory, returns, advantages):
        
        returns = returns.float().to(self.device) 
        actions = torch.tensor(memory.actions).float().to(self.device) 
        #print(memory.observations)
        observations = torch.tensor(memory.observations).float().to(self.device).detach()
        advantages = advantages.float().to(self.device).detach()
        old_logprobs =  torch.stack(memory.logprobs).float().to(self.device).detach()

        # repeat epochs
        for ep in range(self.epochs):
            
            # Batch updates
            permutation = torch.randperm(returns.size()[0])
            for i in range(0,returns.size()[0], self.batch_size):
                indices = permutation[i:i+self.batch_size]

                batch_observations = observations[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                logprobs, state_values, dist_entropy = self.actorcritic.evaluate(batch_observations, batch_actions)

                batch_oldlogprobs = old_logprobs[i:i+self.batch_size]
                ratios = torch.exp(logprobs - batch_oldlogprobs.detach())

                batch_advantages = advantages[i:i+self.batch_size]
                p1 = ratios*batch_advantages
                p2 =  torch.clamp(ratios, 1-self.eps_clipping, 1+self.eps_clipping)*batch_advantages
                
                L1 = -torch.min(p1, p2)

                batch_returns = returns[i:i+self.batch_size]
                L2 = self.MseLoss(state_values, batch_returns)

                L3 = -dist_entropy
                
                loss = L1 + self.c1*L2 + self.c2*L3
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        return loss.cpu().detach().numpy(), L1.cpu().detach().numpy(), L3.cpu().detach().numpy()


    def test(self):
        env_test = gym.make(self.config['env'])
        observation = env_test.reset()
        observation = torch.from_numpy(observation).float().to(self.device)
        reward_episode = 0
        done = False
        with torch.no_grad():
            while not done:
                action = int(self.actorcritic.select_action(observation, store=False))

                observation, reward, done, info = env_test.step(action)
                observation = torch.from_numpy(observation).float().to(self.device)
                reward_episode += reward
            
        env_test.close()

        return reward_episode