import gym
import torch
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse

from networks import CustomDiscreteActorNetwork
from utils import get_gif
import pickle


parser = argparse.ArgumentParser()

parser.add_argument('--path_to_instance',
                    help='where is the instance you want to test',
                    default="CartPole-v1_89957121")
parser.add_argument('--episodes',
                    help='number of games for evaluation', default=100, type=int)

parser.add_argument('--render', action='store_true',
                    help='if true render the games', default=0)

parser.add_argument('--get_gif', action='store_true',
                    help='if true save a game as a gif', default=0)

args = parser.parse_args()

path_instance = os.path.join("experiences",args.path_to_instance)
episodes = args.episodes
name_env = args.path_to_instance.split("_")[0]

print()
print("Evaluating experiment {} on {} games ...\n".format(args.path_to_instance, args.episodes), "\n")

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
            if args.render:
                env.render()
            observation = torch.from_numpy(observation).float()
            action = policy.select_action(observation)[0]
            observation, reward, done, info = env.step(action)
            score += reward

            if done or t == 999:
                list_scores += [score]
                break
    final_res += [list_scores]

if args.get_gif:
    print()
    print("Saving GIF ...")
    get_gif(path_instance, name_env, "A2C_loss_actor")
print()
print("Results: \n")
df = pd.DataFrame(np.array(final_res).T)
df.columns = ["clipped_loss", "adaptative_KL_loss", "A2C_loss"]
print(df.describe().to_string())
pickle.dump(df,open(os.path.join("experiences",args.path_to_instance,"logs","eval_results.pkl"),"wb"))
print("saved at {}".format(os.path.join("experiences",args.path_to_instance,"logs","eval_results.pkl")))

