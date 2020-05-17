import gym
from networks import CustomDiscreteActorNetwork
import torch
from matplotlib import animation
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment name',
                    default='CartPole-v1')
parser.add_argument('--path_to_instance',
                    help='where is the instance you want to test',
                    default="experiences\\CartPole-v1_2199621")
parser.add_argument('--loss',
                    help='discount rate', default="A2C_loss_actor")

args = parser.parse_args()


def save_frames_as_gif(frames, path=r'.\gif_', filename='gym_animation.gif'):

    plt.figure(figsize=(frames[0].shape[1] / 72.0,
                        frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate,
                                   frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def get_gif(path_to_instance,
            name_env='CartPole-v1',
            loss="A2C_loss_actor"):
    env = gym.make(name_env)
    policy = CustomDiscreteActorNetwork(env.observation_space.shape[0],
                                        64,
                                        env.action_space.n)

    path = os.path.join(path_to_instance, "weights", loss + ".pth")
    policy.load_state_dict(torch.load(path))
    policy.eval()
    observation = env.reset()
    frames = []
    score = 0
    for t in range(1000):

        frames.append(env.render(mode="rgb_array"))
        observation = torch.from_numpy(observation).float()
        action = policy.select_action(observation)[0]
        observation, reward, done, info = env.step(action)
        score += reward

        if done or t == 999:
            print("Episode finished after {} timesteps".format(t+1))
            print("Score :" + str(score))
            break
    name = name_env + "_" + loss + ".gif"
    env.close()
    print("----gif processing----")
    save_frames_as_gif(frames, filename=name)


get_gif(args.path_to_instance, args.env, args.loss)
