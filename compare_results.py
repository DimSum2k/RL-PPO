import gym
from networks import CustomDiscreteActorNetwork
import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment name',
                    default='CartPole-v1')
parser.add_argument('--path_to_instance',
                    help='where is the instance you want to test',
                    default="experiences\\CartPole-v1_2199621")
parser.add_argument('--episodes',
                    help='discount rate', default=10)

args = parser.parse_args()

path_instance = args.path_to_instance
episodes = args.episodes
name_env = args.env

final_res = []
for loss in ["clipped_loss_actor",
             "adaptative_KL_loss_actor",
             "A2C_loss_actor"]:
    print("-----------------"+loss+"-----------------")
    env = gym.make(name_env)
    policy = CustomDiscreteActorNetwork(env.observation_space.shape[0],
                                        64,
                                        env.action_space.n)

    path = os.path.join(path_instance, "weights", loss + ".pth")
    policy.load_state_dict(torch.load(path))
    policy.eval()
    list_scores = []
    for i_episode in tqdm(range(episodes)):

        observation = env.reset()
        score = 0
        for t in range(1000):
            env.render()
            observation = torch.from_numpy(observation).float()
            action = policy.select_action(observation)[0]
            observation, reward, done, info = env.step(action)
            score += reward

            if done or t == 999:
                list_scores += [score]
                break
    final_res += [list_scores]

df = pd.DataFrame(np.array(final_res).T)
df.columns = ["clipped_loss", "adaptative_KL_loss", "A2C_loss"]
print(df.describe().to_string())
