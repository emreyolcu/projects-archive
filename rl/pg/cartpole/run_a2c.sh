#!/usr/bin/env bash

python a2c.py \
       --seed 1 --env_name CartPole-v0 --discount 0.99 --solve_threshold 198 \
       --train_episodes 1500 --report_freq 5 --test_freq 5 --test_episodes 100 \
       --actor_learning_rate 0.001 --critic_learning_rate 0.001 --N 100 --optimizer Adam \
       --reward_scale 0.1 --plot_filename cartpole/figure_a2c.png \
       "$@"
