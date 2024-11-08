#!/usr/bin/env bash

python dqn.py \
       --seed 1 --log_level info --env_name SpaceInvadersNoFrameskip-v4 --discount 0.99 \
       --train_episodes 10000 --report_freq 100 --epsilon_final 0.01 --epsilon_decay_iter 5000000 \
       --replay_size 1000000 --replay_start_size 50000 \
       --network atari --gpu --n_hidden 256 --learning_rate 1e-4 --async_update_freq 4 \
       --target_update_freq 10000 --ddqn --dueling --loss huber \
       --test_freq 100 --test_episodes 20 --gradient_clip 10 --interpolation area \
       --plot_filename spaceinvaders/figure.png --model_prefix spaceinvaders/model_ \
       "$@"
