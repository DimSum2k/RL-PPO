import seaborn as sns
from pprint import pprint
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env',help='environment name',default='CartPole-v1')

    parser.add_argument('--std',type=float, help='constant std for continuous action space', default=0.001)
    parser.add_argument('--gamma',type=float, help='discount rate', default=0.99)
    parser.add_argument('--lambd',type=float, help='gae parameter', default=1)
    parser.add_argument('--eps_clipping',type=float, help='clipping parameter for ppo clipped loss', default=0.2)
    parser.add_argument('--d_targ',type=float, help='target ppo KL loss', default=0.01)
    parser.add_argument('--beta_KL',type=float, help='initialisation ppo KL loss', default=3)
    #parser.add_argument('--c1',type=float, help='value function weight if shared actor-critic architecture', default=1)
    parser.add_argument('--c2',type=float, help='entropy weight parameter', default=1e-3)

    parser.add_argument('--lr',type=float, help='learning rate actor-critic', default=1e-3)
    parser.add_argument('--hidden_size',type=float, help='number of hidden units actor-critic', default=1e-3)

    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--max_episodes', type=int, help='maximum number of episodes', default=1000)
    parser.add_argument('--max_steps', type=int, help='maximum number of steps per episode', default=300) # A CHECKER
    parser.add_argument('--optimize_every', type=int, help='size of trajectories we learn on', default=2048)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=4)
    #config["reward_norm"]=False # reward normalisation
    #config["randomize_batch"]=False 

    #config['loss_name'] = ["A2C_loss","adaptative_KL_loss","clipped_loss"][2]
    #config['color'] = {"A2C_loss":sns.color_palette("Set2")[0],"adaptative_KL_loss":sns.color_palette("Set2")[1],"clipped_loss":sns.color_palette("Set2")[2]}
    #config["solved_reward"] = {'LunarLander-v2':230,
    #                          'MountainCarContinuous-v0':300,
    #                          'CartPole-v1':300,
    #                          'MountainCar-v0':300}

    #parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    #load, input, save configurations:
    #parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    #parser.add_argument('--netD', default='', help="path to netD (to continue training)")

    return parser





def reset_config(env_name = 'CartPole-v1', print_=False):

    config = {}
    assert env_name in ['CartPole-v1','MountainCarContinuous-v0','LunarLander-v2','MountainCar-v0']
    config['env'] = env_name

    config['std'] = 0.001  # constant std for continuous action space 
    config['gamma'] = 0.99 # Discount rate
    config['lambda'] = 1   # Generalized Advantage Estimator parameter
    config['value_network'] = {'lr': 1e-3}
    config['actor_network'] = {'lr': 1e-3}
    config['eps_clipping'] = 0.2 #range : 0.1-0.3
    config['d_targ'] = 0.01
    config['beta_KL'] = 3
    config['c1'] = 1    # parameter of the value function loss
    config['c2'] = 1e-3 # entropy parameter --> 1e-4 to 1e-2
    config["reward_norm"]=False # reward normalisation

    config['epochs'] = 4 # number of epochs per trajectory
    config['max_episodes'] = 1000
    config['max_steps'] = 300
    config['optimize_every'] = 2048
    config['batch_size'] = 2048 #512-5120 for continuous / 32-512 for discrete
    config["randomize_batch"]=False 
    config['seed'] = 42

    config['loss_name'] = ["A2C_loss","adaptative_KL_loss","clipped_loss"][2]
    config['color'] = {"A2C_loss":sns.color_palette("Set2")[0],"adaptative_KL_loss":sns.color_palette("Set2")[1],"clipped_loss":sns.color_palette("Set2")[2]}
    config["solved_reward"] = {'LunarLander-v2':230,
                              'MountainCarContinuous-v0':300,
                              'CartPole-v1':300,
                              'MountainCar-v0':300}



    ## ?
    config["reset_val"] = None # use to reset the environment with a custom value


    if print_== True :
        print("Training config : \n")
        pprint(config)
    return config