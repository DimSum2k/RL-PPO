import seaborn as sns
from pprint import pprint
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', help='environment name',
                        default='CartPole-v1')
    parser.add_argument('--std',
                        type=float,
                        help='constant std for continuous action space',
                        default=0.001)
    parser.add_argument('--gamma',
                        type=float, help='discount rate', default=0.99)
    parser.add_argument('--lambd', type=float, help='gae parameter', default=1)
    parser.add_argument('--eps_clipping', type=float,
                        help='clipping parameter for ppo clipped loss (range : 0.1-0.3)',
                        default=0.2)
    parser.add_argument('--d_targ',
                        type=float, help='target ppo KL loss', default=1)
    parser.add_argument('--beta_KL', type=float,
                        help='initialisation ppo KL loss', default=3)
    parser.add_argument('--c1', type=float,
                        help='value function weight if shared actor-critic architecture',
                        default=0)
    parser.add_argument('--c2', type=float,
                        help='entropy weight parameter', default=1e-3)
    parser.add_argument('--lr', type=float,
                        help='learning rate actor-critic', default=1e-3)
    parser.add_argument('--hidden_size',
                        type=float, help='number of hidden units actor-critic',
                        default=1e-3)
    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--max_episodes', type=int,
                        help='maximum number of episodes', default=1200)
    parser.add_argument('--max_steps', type=int,
                        help='maximum number of steps per episode',
                        default=300)
    parser.add_argument('--optimize_every', type=int,
                        help='size of trajectories we learn on', default=2048)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--reward_norm', action='store_true',
                        help='if true normalise rewards', default=0)
    return parser


def reset_config(opt, print_=False):

    config = {}
    assert opt.env in ['CartPole-v1', 'MountainCarContinuous-v0',
                       'LunarLander-v2', 'MountainCar-v0']
    config['env'] = opt.env
    config['std'] = opt.std
    config['gamma'] = opt.gamma
    config['lambda'] = opt.lambd
    config['lr'] = opt.lr
    config['eps_clipping'] = opt.eps_clipping
    config['d_targ'] = opt.d_targ
    config['beta_KL'] = opt.beta_KL
    config['c1'] = opt.c1
    config['c2'] = opt.c2
    config["reward_norm"] = opt.reward_norm
    config['epochs'] = opt.epochs
    config['max_episodes'] = opt.max_episodes
    config['max_steps'] = opt.max_steps
    config['optimize_every'] = opt.optimize_every
    config['batch_size'] = opt.batch_size
    config['seed'] = opt.seed

    config['loss_name'] = ["A2C_loss", "adaptative_KL_loss", "clipped_loss"][2]
    config['color'] = {"A2C_loss": sns.color_palette("Set2")[0],
                       "adaptative_KL_loss": sns.color_palette("Set2")[1],
                       "clipped_loss": sns.color_palette("Set2")[2]}
    config["solved_reward"] = {'LunarLander-v2': 230,
                               'MountainCarContinuous-v0': 300,
                               'CartPole-v1': 300,
                               'MountainCar-v0': 300}

    if print_:
        print("Training config : \n")
        pprint(config)
    return config
