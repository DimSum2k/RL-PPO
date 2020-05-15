import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os

def plot_result(*df,config,save,var=["Episode","Reward"]):
    plt.figure(figsize=(8,4))
    for r in df :
        loss_name = r['loss_name'].unique()[0]
        sns.lineplot(x=var[0], y=var[1],  ci='sd', data=r, color=config["color"][loss_name],label=loss_name)

    plt.savefig(os.path.join(save,"images","rewards.png")) 

def plot_sumup(rewards_list,config,save, loss_list=None):
    plot_result(*rewards_list,config=config,save=save)

    if loss_list is not None:
        plot_result(*loss_list,config=config,var=["Update","entropy"])
        plot_result(*loss_list,config=config,var=["Update","dry_loss"])

def plot_sensitivity(*df,config,label_list,var=["Episode","Reward"]):
    plt.figure(figsize=(8,4))
    for i in range(len(label_list)):
        r=df[i]
        col = list(sns.color_palette("Set1")+sns.color_palette("Set3"))[i]
        sns.lineplot(x=var[0], y=var[1],  ci='sd', data=r, 
                     color=col,label=label_list[i])



def create_folders(env_name):

    path = "./experiences"
    name = env_name + "_" + str(np.random.randint(0,1e8))

    try:
        os.mkdir(os.path.join(path,name))
        os.mkdir(os.path.join(path,name,"images"))
        os.mkdir(os.path.join(path,name,"weights"))
        os.mkdir(os.path.join(path,name,"logs"))
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

    return os.path.join(path,name)



welcome =  """
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