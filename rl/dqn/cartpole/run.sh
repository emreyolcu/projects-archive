#!/usr/bin/env bash

python dqn.py \
       --seed 1 --env_name CartPole-v0 \
       --train_episodes 150 --report_freq 5 --test_freq 5 --test_episodes 100 \
       --test_epsilon 0 --learning_rate 1e-4 --activation relu --target_update_freq 100 \
       --n_mlp_layer 3 --n_hidden 128 --epsilon_decay_iter 10000 --dueling --plot_filename cartpole/figure.png --model_prefix cartpole/model_ \
       "$@"
