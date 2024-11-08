#!/usr/bin/env bash

python dqn.py \
       --seed 1 --env_name MountainCar-v0 --discount 1.0 \
       --train_episodes 1200 --report_freq 25 --test_freq 25 --test_episodes 100 \
       --test_epsilon 0 --learning_rate 1e-4 --activation relu --target_update_freq 100 \
       --n_mlp_layer 3 --n_hidden 128 --epsilon_decay_iter 20000 --epsilon_init 0.2 --epsilon_final 0.05 \
       --dueling \
       --plot_filename mountaincar/figure.png --model_prefix mountaincar/model_ \
       "$@"
