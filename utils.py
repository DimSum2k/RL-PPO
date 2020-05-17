import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import gym
from networks import CustomDiscreteActorNetwork
import torch
from matplotlib import animation
import sys


def plot_result(*df, config, save, var=["Episode", "Reward"]):
    plt.figure(figsize=(8, 4))
    for r in df:
        loss_name = r['loss_name'].unique()[0]
        sns.lineplot(x=var[0], y=var[1],
                     ci='sd', data=r,
                     color=config["color"][loss_name], label=loss_name)

    plt.savefig(os.path.join(save, "images", "rewards.png"))


def plot_sumup(rewards_list, config, save, loss_list=None):
    plot_result(*rewards_list, config=config, save=save)

    if loss_list is not None:
        plot_result(*loss_list, config=config, var=["Update", "entropy"])
        plot_result(*loss_list, config=config, var=["Update", "dry_loss"])


def plot_sensitivity(*df, config, label_list, var=["Episode", "Reward"]):
    plt.figure(figsize=(8, 4))
    for i in range(len(label_list)):
        r = df[i]
        col = list(sns.color_palette("Set1")+sns.color_palette("Set3"))[i]
        sns.lineplot(x=var[0], y=var[1],  ci='sd', data=r,
                     color=col, label=label_list[i])


def create_folders(env_name):

    path = "./experiences"
    name = env_name + "_" + str(np.random.randint(0, 1e8))

    try:
        os.mkdir(os.path.join(path, name))
        os.mkdir(os.path.join(path, name, "images"))
        os.mkdir(os.path.join(path, name, "weights"))
        os.mkdir(os.path.join(path, name, "logs"))
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return os.path.join(path, name)


def save_frames_as_gif(frames, filename):

    plt.figure(figsize=(frames[0].shape[1] / 72.0,
                        frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate,frames=len(frames), interval=50)
    anim.save(filename, writer='imagemagick', fps=60)


def get_gif(path_to_instance, name_env='CartPole-v1', loss="A2C_loss_actor"):

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

    name = os.path.join(path_to_instance, "images", loss + ".gif")
    save_frames_as_gif(frames, filename=name)
    print("Gif saved at {}".format(name))
    env.close()




welcome = """
   \\ \\      / /__| | _ _  _ _ __   _
    \\ \\ /\\ / / _ \\ |/ _/ _ \\| ' ` _ \\ / _ \\
     \\ V  V /  _/ | (_| () | | | | | |  __/
      \\_/\\_/ \\___|_|\\___\\___/|_| |_| |_|\\___|
    \n
    ________________________________________________
    < Welcome ! Let's do Reinforcement Learning ! >
    --------------------------------------------------
    \\   ^__^
     \\  (oo)\\_______
      //(__)\\       )\\/\\
             ||----w |
             ||     ||"""
