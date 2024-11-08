#!/usr/bin/env bash

python a2c.py \
       --seed 1 --env_name LunarLander-v2 --discount 0.99 --solve_threshold 220 \
       --train_episodes 50000 --report_freq 50 --test_freq 100 --test_episodes 100 \
       --actor_learning_rate 0.001 --critic_learning_rate 0.001 --N 100 --optimizer Adam \
       --reward_scale 0.01 --plot_filename lunarlander/figure_a2c.png \
       "$@"
