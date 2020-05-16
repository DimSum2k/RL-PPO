import gym
from networks import CustomDiscreteActorNetwork
import torch
from time import time

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, './videos/' + str(time()) + '/', force = True)

policy = CustomDiscreteActorNetwork(env.observation_space.shape[0],
                                    64,
                                    env.action_space.n)

path = "experiences\CartPole-v1_2199621\weights\clipped_loss_actor.pth"
policy.load_state_dict(torch.load(path))
policy.eval()

for i_episode in range(1):

    observation = env.reset()  # reset for each new trial

    for t in range(1000):  # run for 100 timesteps or until done, whichever is first
        env.render()
        observation = torch.from_numpy(observation).float()
        action = policy.select_action(observation)[0]
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
gym.upload('C:\\Users\\delan\\Documents\\RL-PPO\\videos\\1589642668.3577478')
