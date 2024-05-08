# PPO with curiosity learning

This repository contains rought and low level implementaion of PPO with curiosity learning. This repo is code for paper I wrote recently ![Analiza algorytmów uczenia przez wzmacnianie z zaimplementowanym algorytmem PPO (Proximal Policy Optimization)](https://cris.pk.edu.pl/info/article/CUTb91a6b74ee7b4b5bae4c4367cfdb0489/)

## MsPacman model sample

![Sample animation](animations/MsPacman-v5v6.1_20231214-094337_3000-tryhard.gif)

It is easly able to achive score `>3000` in pacman. In comparsion with DQN I was able to achive at most 700 points on avg.  

![Avg score Pacman](plots/all_models.png)

To reproduce effects run appropriate train script (`src/train_*.pl`). Hyperparameters were changed during training. For exac values, plese refer to linked paper.   
