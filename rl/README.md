# Deep RL

This directory includes implementations of deep reinforcement learning
algorithms in PyTorch.

## Requirements

In order to install the required libraries it is recommended that you create a
conda environment by running

    conda env create -f environment.yaml

and before running any experiments activate the environment:

    source activate rl

## Running

With each implementation there are scripts included for several OpenAI Gym
environments. For instance, training a DQN for CartPole with the default
configuration requires you to run the following.

    cd dqn
    ./cartpole/run.sh
