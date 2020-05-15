import matplotlib.pyplot as plt
import seaborn as sns

def plot_result(*df,config,var=["Episode","Reward"]):
    plt.figure(figsize=(8,4))
    for r in df :
        loss_name = r['loss_name'].unique()[0]
        sns.lineplot(x=var[0], y=var[1],  ci='sd', data=r, color=config["color"][loss_name],label=loss_name) 
def plot_sumup(rewards_list,loss_list,config):
    plot_result(*rewards_list,config=config)
    plot_result(*loss_list,config=config,var=["Update","entropy"])
    plot_result(*loss_list,config=config,var=["Update","dry_loss"])

def plot_sensitivity(*df,config,label_list,var=["Episode","Reward"]):
    plt.figure(figsize=(8,4))
    for i in range(len(label_list)):
        r=df[i]
        col = list(sns.color_palette("Set1")+sns.color_palette("Set3"))[i]
        sns.lineplot(x=var[0], y=var[1],  ci='sd', data=r, 
                     color=col,label=label_list[i])