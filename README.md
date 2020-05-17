# RL-PPO: Proximal Policy Optimization Algorithms: a review

***Authors:*** Dimitri Meunier, Philomène Chagniot and Pierre Delanoue

This repository contains the code and materials used to obtain the results of our review of the paper _Proximal Policy Optimization Algorithms_ by Schulman et al.

This project was the final evaluation of the course of Reinforcement Learning taught by Moez Draief for the 3rd year of the engineering cycle at ENSAE Paris.

***Usage***

After having clone this repository, if you want to replicate the traning, you need to run the _main_train.py_ code:

``` python main_train.py ```

The path to your simulation instance will be printed at the end (ex: experiences\CartPole-v1_2199621). 
You can now benchmark and evaluate the different losses with _main_test.py_. If you want to evaluate the pretrained network run:

``` python main_test.py ```

If you want to evaluate you own results use your simulation instance (ex: experiences\CartPole-v1_2199621).

``` python main_test.py --instance "CartPole-v1_2199621" ```

You can also capture a gif of a game by adding _--get_gif_: 

``` python main_test.py --get_gif```

If you want to display the games you evaluate on use (add _--episode_ to reduce the number of games as the rendering is slow):

``` python main_test.py --render --episodes 5```

***Advantage Actor-Critic on CartPole v1***: 
Display of an Advantage Actor Critic policy trained for Cartpole

<img width="500px" src="gif/CartPole_A2C.gif">

