from config import get_arguments, reset_config
from ppo import PPOAgent
from utils import plot_sumup, create_folders, welcome

import torch
import pprint
import os
import pickle as pkl
import time

if __name__ == '__main__':

    print(welcome)
    time.sleep(2)

    parser = get_arguments()
    opt = parser.parse_args()

    folder_name = create_folders(opt.env)

    rewards_list = []
    loss_list = []

    for loss in ["clipped_loss", "adaptative_KL_loss", "A2C_loss"]:
        print("-----------------"+loss+"-----------------")
        config = reset_config(opt, print_=False)

        if loss == "A2C_loss":
            config["batch_size"] = 128
            config['epoch'] = 1
            config["c2"] = 0

        if loss == "clipped_loss":
            config['epoch'] = 8

        config["loss_name"] = loss

        agent = PPOAgent(config)

        rewards, Loss = agent.training(config["epochs"],
                                       config["optimize_every"],
                                       config["max_episodes"],
                                       config["max_steps"])
        rewards_list.append(rewards)
        loss_list.append(Loss)

        # save models
        torch.save(agent.actor_network.state_dict(), os.path.join(folder_name,
                                                                  "weights",
                                                                  loss + "_actor.pth"))
        torch.save(agent.value_network.state_dict(),
                   os.path.join(folder_name, "weights", loss + "_critic.pth"))

    plot_sumup(rewards_list, config=config, save=folder_name)
    f = open(os.path.join(folder_name, "logs", "config.txt"), "w")
    f.write(pprint.pformat(config))
    f.close()
    pkl.dump({"config": config, "loss": loss_list, "rewards": rewards_list},
             open(os.path.join(folder_name, "logs", "losses_rewards_config.pkl"),
             "wb"))
    print()
    print("Simulation saved at {}".format(folder_name))
