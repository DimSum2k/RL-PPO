
import matplotlib.pyplot as plt

### Note : functions defined in https://sites.google.com/view/rlensae2020/hands-on?authuser=0

def moving_average(values, window):
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)

def plot_steps_per_episode(number_steps_per_episode, smoothing=8):
    f1 = plt.figure(1, figsize=(12, 7))
    plt.title(" Title : Agent's number of actions per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Number of actions")
    plt.plot(number_steps_per_episode, label='raw plot')
    plt.plot(moving_average(number_steps_per_episode, smoothing),
             label='moving average')
    plt.legend()
    f1.show()
    
def plot_average_reward(average_reward_per_episode, smoothing=10):
    f2 = plt.figure(2, figsize=(12, 7))
    plt.title(" Title : Agent's average reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Average reward")
    plt.plot(average_reward_per_episode, label='raw plot')
    plt.plot(moving_average(average_reward_per_episode, smoothing),
             label='moving average')
    plt.legend()
    f2.show()

def plot_total_reward(total_reward_per_episode, smoothing=10):
    f3 = plt.figure(3, figsize=(12, 7))
    plt.title(" Title : Agent's total reward per episode")
    plt.xlabel("Episodes")
    plt.ylabel("Total reward")
    plt.plot(total_reward_per_episode, label='raw plot')
    plt.plot(moving_average(total_reward_per_episode, smoothing),
             label='moving average')
    plt.legend()
    f3.show()
    
def plot_episode_duration(episodes_durations, smoothing=10):
    f4 = plt.figure(4, figsize=(12, 7))
    plt.title(" Title : Duration of each episode")
    plt.xlabel("Episodes")
    plt.ylabel("Duration")
    plt.plot(episodes_durations, label='raw plot')
    plt.plot(moving_average(episodes_durations, 10), label='moving average')
    plt.legend()
    f4.show()
    
    