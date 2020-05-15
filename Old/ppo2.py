import pandas as pd
import itertools
import numpy as np
import datetime
import gym
from gym.wrappers import Monitor
from gym.spaces import Box, Discrete

import torch 
import torch.nn as nn
from torch import optim
import torch.nn.functional as F 
##from torch.distributions import normal

from memory import Memory
from networks import CustomValueNetwork, CustomDiscreteActorNetwork

class PPOAgent:

    def __init__(self, config):
        
        self.config = config
        self.memory = Memory()
        self.device='cpu'
        self.env = gym.make(config['env'])
        
        # boolean for discrete action space:
        self.discrete_action_bool = isinstance(self.env.action_space, Discrete) 
        self.gamma = config['gamma'] 
        self.lambd = config['lambda'] 
        self.c1 = config['c1'] 
        self.c2 = config['c2'] 
        self.norm_reward = config["reward_norm"]
        self.loss_name = config['loss_name'] 
        self.kl_div = nn.KLDivLoss(reduction="none") 
        self.beta_kl = config['beta_KL']
        # specify a value for env reset ???
        self.reset_val = config["reset_val"] 
        self.batch_size = config["batch_size"]
        if self.discrete_action_bool == False :
            print("Low : ",self.env.action_space.low)
            print("High : ",self.env.action_space.high)
            
        # set random seeds
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        self.env.seed(config['seed'])
        
        # Critic
        self.value_network = CustomValueNetwork(self.env.observation_space.shape[0], 64, 1).to(self.device)
        self.value_network_optimizer: optim.Optimizer = optim.Adam(
            self.value_network.parameters(), lr=config['value_network']['lr'])
            
        # Actor     
        if self.discrete_action_bool :
            self.actor_network = CustomDiscreteActorNetwork(self.env.observation_space.shape[0], 64, self.env.action_space.n).to(self.device)
        else :
            self.actor_network = ContinuousActorNetwork(self.env.observation_space.shape[0], 64, self.env.action_space.shape[0], self.config["std"], self.env).to(self.device)
        
        self.actor_network_optimizer: optim.Optimizer = optim.Adam(
            self.actor_network.parameters(), lr=config['actor_network']['lr'])
        
        # save in memory policy estimates
        self.probs_list = []  # probability of actions taken
        self.mean_list = []     # mean estimate (for continuous action)
        
    def _returns_advantages(self, values, next_value):
        """Returns the cumulative discounted rewards with GAE

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
        
        rewards = np.array(self.memory.rewards)
        if self.norm_reward:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        returns, advantages = [], []
        last = next_value
        gae = 0 

        for i in reversed(range(len(self.memory))):
            # build the returns
            returns.insert(0,rewards[i] + self.gamma*last*(1-self.memory.dones[i]))
            # build the advantages
                #delta = returns[0] - self.memory.values[i]
            delta = rewards[i] + self.gamma*next_value*(1-self.memory.dones[i]) - values[i]
            gae = delta + self.gamma*self.lambd*(1-self.memory.dones[i])*gae
            #returns.insert(0,gae + self.memory.values[i])
            advantages.insert(0,gae)
            #next_value = self.memory.values[i]
            next_value = values[i]

        returns = torch.FloatTensor(returns).to(self.device)
        #advantages = returns - torch.FloatTensor(self.memory.values).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        return returns, advantages 

    
    
    
    def training(self, epochs, optimize_every, max_episodes, max_steps):
        t1 = datetime.datetime.now()
        """Perform a training by batch
            Parameters
            ----------
            epochs : int
                Number of epochs
            batch_size : int\n",
                The size of a batch"""
        
        episode_count = 0
        timestep_count = 0
        rewards_test = []
        
        loss_evol = {'loss':[],'dry_loss':[],'entropy':[]}
        if self.loss_name not in ["A2C_loss","adaptative_KL_loss","clipped_loss"]:
            print('Unknown loss function, using clipped loss as default loss')
        else :
            print('Loss : ',self.loss_name)
    
        for ep in range(max_episodes):
            episode_count +=1
            #assert self.break_bool == False
            obs = self.env.reset()
            ## added : special initialization
            #if self.reset_val is not None:\n",
            #self.env.env.state = np.array(self.reset_val)
            #obs = self.env.env.state
                                  
            for i in range(max_steps):
                timestep_count +=1
                self.memory.observations.append(obs)  # just observed s_t
                obs_t = torch.from_numpy(obs).float().to(self.device)  # tensor
                action = self.actor_network.select_action(obs_t)  # act on just observed, action a_t
                                  
                if self.discrete_action_bool : #convert to integer only for discrete action space
                    action = int(action)
                self.memory.actions.append(action)
                
                # value function estimate of s_t
                #self.memory.values.append(self.value_network(obs_t))
                                  
                ## Run a step : get new state s_{t+1} and rewards r_t
                obs, reward, done, _ = self.env.step(action) 

                # Store termination status reward
                self.memory.dones.append(done)
                self.memory.rewards.append(reward)

                if (timestep_count % optimize_every) == 0 :
                    # Learning step !
                    #if done :
                    #    next_value = 0
                    #else:
                    #    obs_t = torch.from_numpy(self.memory.observations[-1]).float().to(self.device)
                    #    next_value = self.value_network.predict(obs_t)
                                  
                    # Compute returns and advantages
                    #returns_, advantages_ = self._returns_advantages(next_value)
                    #returns.append(returns_)
                    #advantages.append(advantages_)

                    for epoch in range(epochs):
                        #loss_val, dry_loss_val, entrop_val = self.optimize_model(returns_, advantages_)
                        ##loss_val, dry_loss_val, entrop_val = self.optimize_model(obs, epoch)
                        loss_val, dry_loss_val, entrop_val = self.optimize_model(obs)
                        # only save result of last epoch #?? save more for fine tuning nb of epochs ??
                        if epoch == epochs-1 : 
                            loss_evol["loss"].append(loss_val)
                            loss_evol["dry_loss"].append(dry_loss_val)
                            loss_evol["entropy"].append(entrop_val)

                    self.memory.clear_memory()

                if done:
                    break 

            # Test every 25 episodes
            if ep == 1 or (ep > 0 and ep % 25 == 0) or (ep == max_episodes - 1):
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                print(f'Episode {ep}/{max_episodes}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}')

            #?? Early stopping ++ to add --> which condition ?? certain avg reward ?
        self.env.close()
        t2 = datetime.datetime.now()

        # save rewards
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Episode', 'Reward'])
        r["Episode"] = r["Episode"]*25
        r["loss_name"] = self.loss_name # add loss name as label

#         # Plot\n",
#         sns.lineplot(x=\"Episode\", y=\"Reward\", data=r, ci='sd',color=config[\"color\"][self.loss_name],label=self.loss_name)\n",

        # Total time ellapsed\n",
        time = t2-t1
        print(f'The training was done over a total of {episode_count} episodes')
        print('Total time ellapsed during training : ',time)
        r["time"]=time
        loss_evol = pd.DataFrame(loss_evol).astype(float)
        loss_evol["loss_name"] = self.loss_name
        loss_evol["Update"] = range(len(loss_evol))
        return r, loss_evol


        
    def A2C_loss(self, prob, actions, advantages):
        loss = 0.
        if self.discrete_action_bool :
            for i in range(len(actions)):
            	loss -= torch.log(prob[i, int(actions[i])]+1e-6)*advantages[i]
            ##loss = torch.log(prob.gather(1, actions.long().view(-1,1))+1e-6)*advantages
            ##loss = -torch.tensor([torch.log(prob[i, int(actions[i])]+1e-6)*advantages[i] for i in range(len(actions))])
        else :
            loss = torch.dot(torch.log(prob.view(-1)+1e-6),advantages)      
        ##return loss*2048
        return loss
    
    
    def compute_proba_ratio(self, prob, actions):
    ##def compute_proba_ratio(self, actions, epoch):
        if self.discrete_action_bool: 
            #1st iteration : initialize old policy to the current one to avoid clipping
            if len(self.probs_list) == 1:
            ##if epoch==0:
                old_prob = self.probs_list[0]
            else:
                old_prob = self.probs_list[len(self.probs_list)-2]
            ##old_prob = self.memory.probs[0]
        else :
            if len(self.mean_list) == 1:
                old_prob_mean = self.mean_list[0]
            else :
                old_prob_mean = self.mean_list[len(self.mean_list)-2]
                
            m = normal.Normal(loc = old_prob_mean.float(), scale = torch.tensor(config["std"]*np.ones(actions.size())).float())
            old_prob = m.log_prob(actions.float()).reshape(actions.size()).detach()
            
        # Discrete action space
        if self.discrete_action_bool :
            # compute the ratio directly using gather function
            num = prob.gather(1, actions.long().view(-1,1))
            ##num = self.memory.probs[1].gather(1, actions.long().view(-1,1))
            denom = old_prob.detach().gather(1, actions.long().view(-1,1))
            ratio_vect = num.view(-1)/denom.view(-1)
        
        # Continuous action space
        else :## NB : add small constant to avoid ratio explosion 
            #?? replace "+1e-6" by clamp to min=1e-6?
            ratio_vect = prob/(old_prob+1e-6)
        
        if np.isnan(ratio_vect.cpu().detach().numpy()).any():
            print("NaN encountered in proba ratio")

        return ratio_vect, old_prob
    

    
    def clipped_loss(self, prob, actions, advantages):##++TODO
    ##def clipped_loss(self, actions, advantages, epoch):
        ##Compute the probability ratio
        ratio_vect = self.compute_proba_ratio(prob, actions)[0]
        ##if epoch==0:
        ##    ratio_vect=torch.zeros_like(actions) + 1
        ##else:
        ##    ratio_vect = self.compute_proba_ratio(actions, epoch)[0]
        if len(actions.size())>1 :
            ratio_vect = torch.prod(ratio_vect, dim = 1)
        
        ## Compute the loss
        loss1 = ratio_vect * advantages
        loss2 = torch.clamp(ratio_vect, 1-self.config['eps_clipping'], 1+self.config['eps_clipping']) * advantages
        loss = - torch.sum(torch.min(loss1, loss2))#/10#self.config["batch_size"]
        ##loss = - torch.min(loss1, loss2)

        return loss

    def adaptative_KL_loss(self, prob, actions, advantages, observations):
        ## Compute the loss
        if self.discrete_action_bool :
            ##Compute the probability ratio
            ratio_vect, old_prob = self.compute_proba_ratio(prob, actions)
            kl = self.kl_div(prob[observations.long(),:], old_prob[observations.long(),:])
            kl = kl.view(prob.size()[0],-1).mean(1)
        else :
            ratio_vect = self.compute_proba_ratio(prob, actions)[0]
            if len(self.mean_list) == 1:
                kl = torch.tensor(0.)
            else :
                mu = prob.view(-1)
                mu_old = self.mean_list[len(self.mean_list)-2].view(-1).detach()
                # use KL formula for gaussian 
                kl = torch.dot((mu-mu_old)/torch.tensor(config["std"]*np.ones(len(actions))).float(),mu-mu_old)/2
                
        #loss = - torch.sum((ratio_vect*advantages) - (self.beta_kl*kl))#/10#self.config["batch_size"]
        loss = - ((ratio_vect*advantages) - (self.beta_kl*kl))
        
        # Update beta values
        if np.isnan(torch.mean(kl).cpu().detach().numpy()):
            print("Nan encountered in average KL divergence")
        if torch.mean(kl) < self.config["d_targ"]/1.5 : 
            self.beta_kl = self.beta_kl / 2
        elif torch.mean(kl) > self.config["d_targ"]*1.5 :
            self.beta_kl = self.beta_kl * 2
        return loss
    
    def optimize_model(self, next_obs):
    ##def optimize_model(self, next_obs, epoch):
        
        losses = {"loss":[],"dry_loss":[],"entropy":[]}
        
        #n_trajs = np.sum(self.memory.dones)
        #permutation = torch.randperm(observations.size()[0]) # /!\ order
        permutation = torch.arange(len(self.memory))
        
        # Obs - act - rew - dones STAY FIX
        #returns = returns[len(returns)-1].float()
        #advantages = advantages[len(advantages)-1].float()
        observations = torch.tensor(self.memory.observations).float().to(self.device)#.detach()#[permutation]
        actions = torch.tensor(self.memory.actions).float().to(self.device)#.detach()#[permutation]
        
        # build advantages and returns 

        next_obs = torch.from_numpy(next_obs).float().to(self.device)#.detach()
        next_value = self.value_network.predict(next_obs)
        values = self.value_network(observations)#.detach()#[permutation]
        returns, advantages = self._returns_advantages(values, next_value)
        returns = returns.float().to(self.device)#.detach()
        advantages = advantages.float().to(self.device)#.detach()

        for i in range(0,returns.size()[0], self.batch_size):
            ##if i==0: epoch=0
            ##else: epoch=1
            
            indices = permutation[i:i+self.batch_size]
            batch_observations = observations[i:i+self.batch_size]
            batch_actions = actions[i:i+self.batch_size]
            batch_returns = returns[i:i+self.batch_size]
            batch_advantages = advantages[i:i+self.batch_size]
            
            
            # Critic loss
            net_values: torch.Tensor = self.value_network(batch_observations)
            critic_loss = F.mse_loss(net_values.view(-1), batch_returns)
            critic_loss.backward()
            self.value_network_optimizer.step()
            
          
            
            # Actor & Entropy loss
            #/!\ doesn't have the same meaning in the discrete and continuous case
            prob: torch.Tensor = self.actor_network(batch_observations) # shape (batch_size,action_space)

            if self.discrete_action_bool == True :
                self.probs_list.append(prob.detach())
                ##self.memory.probs.append(prob.detach())
            else : #/!\ continuous actions may have several dimensions
                m = normal.Normal(loc = prob.float(), scale = torch.tensor(config["std"]*np.ones(actions.size())).float())
                logprob = m.log_prob(actions.float()).reshape(actions.size())
                self.probs_list.append(torch.exp(logprob).detach()) #not very useful 
                # append the gaussian mean (used to estimate old proability)
                self.mean_list.append(prob)
                
            #if not self.memory.dones[-1]:
            #    n_trajs += 1

            if self.loss_name == "clipped_loss":
                loss = self.clipped_loss(prob, batch_actions, batch_advantages)
                ##loss = self.clipped_loss(batch_actions, batch_advantages, epoch)

            elif self.loss_name == "adaptative_KL_loss":
                loss = self.adaptative_KL_loss(prob, batch_actions, batch_advantages, batch_observations)

            elif self.loss_name == "A2C_loss":
                loss = self.A2C_loss(prob, batch_actions, batch_advantages)

            else :#use clipped loss as default
                loss = self.clipped_loss(prob, batch_actions, batch_advantages)
                
            dry_loss = loss 
            entropy_term = -torch.sum(prob * torch.log(prob+1e-6))
            #entropy_term = -torch.sum(prob * torch.log(prob+1e-6), dim=1)
            loss -= (self.c2 * entropy_term)
            ##loss = loss.sum() - (self.c2 * entropy_term)
            #loss = loss / n_trajs

            loss.backward()
            #loss.sum().backward()
            ##loss.mean().backward()
            self.actor_network_optimizer.step()
            
        
            self.value_network_optimizer.zero_grad()
            self.actor_network_optimizer.zero_grad()
            
            losses["loss"].append(loss.mean().item())
            losses["dry_loss"].append(dry_loss.mean().item())
            losses["entropy"].append(entropy_term.mean().item())

        return np.mean(losses["loss"]), np.mean(losses["dry_loss"]), np.mean(losses["entropy"])



    def evaluate(self, render=False):
        env = self.monitor_env if render else self.env
        observation = env.reset()
        ## added : special initialization
        if self.reset_val is not None:
            self.env.env.state = np.array(self.reset_val)
            observation = self.env.env.state
        observation = torch.from_numpy(observation).float().to(self.device)
        reward_episode = 0
        done = False
        with torch.no_grad():
            while not done:
                policy = self.actor_network(observation)
                
                if self.discrete_action_bool : 
                    action = int(torch.multinomial(policy, 1))  # draw an action ##??why not only sample action ?? (coming from Lazaric)
                else :
                    action = self.actor_network.select_action(observation)
                observation, reward, done, info = env.step(action)
                observation = torch.from_numpy(observation).float().to(self.device)
                reward_episode += reward
            
        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
        return reward_episode