import itertools
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import gym
from gym.spaces import Discrete
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
from memory import Memory
from networks import CustomValueNetwork, CustomDiscreteActorNetwork
from networks import ContinuousActorNetwork


class PPOAgent:

    def __init__(self, config):

        self.config = config
        self.memory = Memory()
        self.device = 'cpu'
        self.env = gym.make(config['env'])

        # boolean for discrete action space:
        self.discrete_action_bool = isinstance(self.env.action_space, Discrete)
        self.gamma = config['gamma']
        self.lambd = config['lambda']
        self.c1 = config['c1']
        self.c2 = config['c2']
        self.norm_reward = config["reward_norm"]
        self.loss_name = config['loss_name']
        self.beta_kl = config['beta_KL']

        self.batch_size = config["batch_size"]
        if not(self.discrete_action_bool):
            print("Low : ", self.env.action_space.low)
            print("High : ", self.env.action_space.high)

        # set random seeds
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        self.env.seed(config['seed'])

        # Critic
        self.value_network = CustomValueNetwork(
                            self.env.observation_space.shape[0],
                            64,
                            1).to(self.device)
        self.value_network_optimizer: optim.Optimizer = optim.Adam(
                                            self.value_network.parameters(),
                                            lr=config['lr'])
        # Actor
        if self.discrete_action_bool:
            self.actor_network = CustomDiscreteActorNetwork(
                                        self.env.observation_space.shape[0],
                                        64,
                                        self.env.action_space.n
                                                            ).to(self.device)
        else:
            self.actor_network = ContinuousActorNetwork(
                            self.env.observation_space.shape[0],
                            64,
                            self.env.action_space.shape[0],
                            self.config["std"], self.env
                                                        ).to(self.device)

        self.actor_network_optimizer: optim.Optimizer = optim.Adam(
                                            self.actor_network.parameters(),
                                            lr=config['lr'])

        # save in memory policy estimates
        self.probs_list = []    # probability of actions taken
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
            returns.insert(0, rewards[i] + self.gamma*last*(1-self.memory.dones[i]))

            # build the advantages
            delta = rewards[i] + self.gamma*next_value*(1-self.memory.dones[i]) - values[i]
            gae = delta + self.gamma*self.lambd*(1-self.memory.dones[i])*gae
            advantages.insert(0, gae)
            next_value = values[i]

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        return returns, advantages

    def training(self, epochs, optimize_every, max_episodes, max_steps):
        t1 = datetime.datetime.now()
        """Perform a training by batch
            Parameters
            ----------
            epochs : int
                Number of epochs
            batch_size : int
                The size of a batch"""

        episode_count = 0
        timestep_count = 0
        rewards_test = []
        solved = False

        loss_evol = {'loss': [], 'dry_loss': [], 'entropy': []}
        if self.loss_name not in ["A2C_loss", "adaptative_KL_loss", "clipped_loss"]:
            print('Unknown loss function, using clipped loss as default loss')
        else:
            print('Loss : ', self.loss_name)

        for ep in tqdm(range(max_episodes)):
            if not solved:
                episode_count += 1
                obs = self.env.reset()

                for i in range(max_steps):
                    timestep_count += 1
                    self.memory.observations.append(obs)
                    obs_t = torch.from_numpy(obs).float().to(self.device)
                    action = self.actor_network.select_action(obs_t.view(1, -1))

                    if self.discrete_action_bool:
                        action = int(action)
                        self.memory.actions.append(action)
                        obs, reward, done, _ = self.env.step(action)

                    else:
                        self.memory.actions.append(action)
                        obs, reward, done, _ = self.env.step(action.view(-1))

                    # Store termination status reward
                    self.memory.dones.append(done)
                    self.memory.rewards.append(reward)

                    if (timestep_count % optimize_every) == 0:
                        for epoch in range(epochs):
                            loss_val, dry_loss_val, entrop_val = self.optimize_model(obs)
                            if epoch == epochs-1:
                                loss_evol["loss"].append(loss_val)
                                loss_evol["dry_loss"].append(dry_loss_val)
                                loss_evol["entropy"].append(entrop_val)

                        self.memory.clear_memory()

                    if done:
                        break

            # Test every 25 episodes
            if ep == 1 or (ep > 0 and ep % 25 == 0) or (ep == max_episodes - 1):
                rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
                if round(rewards_test[-1].mean(), 2) == 500.:
                    solved = True

        self.env.close()
        t2 = datetime.datetime.now()

        # save rewards
        r = pd.DataFrame((itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))), columns=['Episode', 'Reward'])
        r["Episode"] = r["Episode"]*25
        r["loss_name"] = self.loss_name

        # Total time ellapsed
        time = t2-t1
        print(f'The training was done over a total of {episode_count} episodes')
        print('Total time ellapsed during training : ', time)
        r["time"] = time
        loss_evol = pd.DataFrame(loss_evol).astype(float)
        loss_evol["loss_name"] = self.loss_name
        loss_evol["Update"] = range(len(loss_evol))

        return r, loss_evol

    def compute_proba_ratio(self, prob, actions):
        if self.discrete_action_bool:
            if len(self.probs_list) == 1:
                old_prob = self.probs_list[0]
            else:
                old_prob = self.probs_list[len(self.probs_list)-2]

        else:
            if len(self.mean_list) == 1:
                old_prob_mean = self.mean_list[0]
            else:
                old_prob_mean = self.mean_list[len(self.mean_list)-2]

            diag = torch.tensor(self.config['std']*np.ones(old_prob_mean.size()[1])).float()
            dist = Normal(old_prob_mean, scale=diag)
            old_prob = dist.log_prob(actions).detach()

            # build new ones
            dist = Normal(prob, scale=diag)
            prob = dist.log_prob(actions)

        if self.discrete_action_bool:
            # compute the ratio directly using gather function
            num = prob.gather(1, actions.long().view(-1, 1))
            denom = old_prob.detach().gather(1, actions.long().view(-1, 1))
            ratio_vect = num.view(-1)/denom.view(-1)

        else:
            if np.isnan(prob.cpu().detach().numpy()).any():
                print("NaN encountered in num ratio")

            if np.isnan(old_prob.cpu().detach().numpy()).any():
                print("NaN encountered in denom ratio")
            ratio_vect = prob/(old_prob+1e-6)

        if np.isnan(ratio_vect.cpu().detach().numpy()).any():
            print("NaN encountered in proba ratio")

        return ratio_vect, old_prob


    def clipped_loss(self, prob, actions, advantages):

        ratio_vect = self.compute_proba_ratio(prob, actions)[0]
        if len(actions.size()) > 1 and not(self.discrete_action_bool):
            ratio_vect = torch.prod(ratio_vect, dim=1)
        loss1 = ratio_vect * advantages
        loss2 = torch.clamp(ratio_vect, 1-self.config['eps_clipping'], 1+self.config['eps_clipping']) * advantages
        loss = - torch.sum(torch.min(loss1, loss2))
        return loss

    def adaptative_KL_loss(self, prob, actions, advantages, observations):
        if self.discrete_action_bool:
            ratio_vect, old_prob = self.compute_proba_ratio(prob, actions)
            kl = torch.zeros(1)
            for i in range(prob.size()[0]):
                kl += (old_prob[i] * (old_prob[i].log() - prob[i].log())).sum()

        else:
            ratio_vect = self.compute_proba_ratio(prob, actions)[0]
            if len(actions.size()) > 1 and not(self.discrete_action_bool):
                ratio_vect = torch.prod(ratio_vect, dim=1)
            if len(self.mean_list) == 1:
                kl = torch.tensor(0.)
            else:
                mu = prob
                mu_old = self.mean_list[len(self.mean_list)-2].detach()
                a = (mu-mu_old)/torch.tensor(config["std"]*np.ones(actions.size())).float()
                b = (mu-mu_old)
                if len(actions.size()) > 1:
                    a = torch.prod(a, axis=1)
                    b = torch.prod(b, axis=1)
                kl = torch.dot(a, b)/2

        loss = - torch.sum((ratio_vect*advantages)) + self.beta_kl*kl
        if np.isnan(torch.mean(kl).cpu().detach().numpy()):
            print("Nan encountered in average KL divergence")
        if kl < self.config["d_targ"]/1.5:
            self.beta_kl = self.beta_kl / 2
        elif kl > self.config["d_targ"]*1.5:
            self.beta_kl = self.beta_kl * 2
        return loss

    def A2C_loss(self, prob, actions, advantages):
        loss = 0.
        if self.discrete_action_bool:
            for i in range(len(actions)):
                loss -= torch.log(prob[i, int(actions[i])]+1e-6)*advantages[i]
        else:
            diag = torch.tensor(self.config["std"]*np.ones(prob.size()[1])).float()
            dist = Normal(prob, scale=diag)
            prob = dist.log_prob(actions)
            if actions.size()[1] > 1:
                prob = torch.prod(prob, dim=1)
            loss = torch.dot(torch.log(prob.view(-1)+1e-6), advantages)

        return loss

    def optimize_model(self, next_obs):

        losses = {"loss": [], "dry_loss": [], "entropy": []}
        idx = torch.arange(len(self.memory))

        observations = torch.tensor(self.memory.observations).float().to(self.device)
        if np.isnan(observations.cpu().detach().numpy()).any():
            print("nan in observations")

        if self.discrete_action_bool:
            actions = torch.tensor(self.memory.actions).float().to(self.device)
        else:
            actions = torch.squeeze(torch.stack(self.memory.actions),1).float().to(self.device)

        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        next_value = self.value_network.predict(next_obs)
        values = self.value_network(observations)
        returns, advantages = self._returns_advantages(values, next_value)
        returns = returns.float().to(self.device)
        advantages = advantages.float().to(self.device)

        for i in range(0, returns.size()[0], self.batch_size):
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
            if np.isnan(batch_observations.cpu().detach().numpy()).any():
                print("nan in batch observations")

            prob: torch.Tensor = self.actor_network.forward(batch_observations)
            if np.isnan(prob.cpu().detach().numpy()).any():
                print("NAN HERE")

            if self.discrete_action_bool:
                self.probs_list.append(prob.detach())

            else:
                self.mean_list.append(prob.detach())

            if self.loss_name == "clipped_loss":
                loss = self.clipped_loss(prob, batch_actions, batch_advantages)

            elif self.loss_name == "adaptative_KL_loss":
                loss = self.adaptative_KL_loss(prob, batch_actions, batch_advantages, batch_observations)

            elif self.loss_name == "A2C_loss":
                loss = self.A2C_loss(prob, batch_actions, batch_advantages)

            else:
                loss = self.clipped_loss(prob, batch_actions, batch_advantages)

            dry_loss = loss
            entropy_term = -torch.sum(prob * torch.log(prob+1e-6))
            loss -= (self.c2 * entropy_term)
            loss.backward()
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

        observation = torch.from_numpy(observation).float().to(self.device)
        reward_episode = 0
        done = False
        with torch.no_grad():
            while not done:
                policy = self.actor_network(observation)

                if self.discrete_action_bool:
                    action = int(torch.multinomial(policy, 1))
                    observation, reward, done, info = env.step(action)
                else:
                    action = self.actor_network.select_action(observation)
                    observation, reward, done, info = env.step(action.view(-1))
                observation = torch.from_numpy(observation).float().to(self.device)
                reward_episode += reward

        env.close()
        if render:
            show_video("./gym-results")
            print(f'Reward: {reward_episode}')
        return reward_episode
