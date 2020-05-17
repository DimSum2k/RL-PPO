# RL-PPO: Proximal Policy Optimization Algorithms: a review

***Authors:*** Dimitri Meunier, Philomène Chagniot and Pierre Delanoue

This repository contains the code and materials used to obtain the results of our review of the paper _Proximal Policy Optimization Algorithms_ by Schulman et al. .
This project was the final evaluation of the course of Reinforcement Learning taught by Moez Draief for the 3rd year of the engineering cycle at ENSAE Paris.

***Usage***

After having clone this repository, you need to run the _main.py_ code:

``` python main.py ```

You can now benchmark the different losses:

``` python compare_results.py --path_to_instance "experiences\CartPole-v1_2199621" ```

You can also capture a gif of a game: 

``` python record_game.py --path_to_instance "experiences\CartPole-v1_2199621" --loss "A2C_loss_actor" ```

***Advantage Actor-Critic on CartPole v1***: 

<img width="500px" src="gif/CartPole_A2C.gif">

