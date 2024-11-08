#!/usr/bin/env bash

python dqn.py \
       --seed 2 --log_level info --env_name PongNoFrameskip-v4 --discount 0.99 \
       --train_episodes 1500 --report_freq 25 --epsilon_final 0.01 \
       --replay_size 10000 --replay_start_size 10000 --clip_reward \
       --network atari --gpu --n_hidden 256 --learning_rate 1e-4 --async_update_freq 4 \
       --target_update_freq 1000 --ddqn --dueling --loss huber \
       --test_freq 25 --test_episodes 20 --gradient_clip 10 --test_epsilon 0.01 --interpolation area --crop \
       --plot_filename pong/figure.png --model_prefix pong/model_ \
       "$@"
