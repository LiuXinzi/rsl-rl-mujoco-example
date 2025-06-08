#!/usr/bin/env bash

python train.py \
    train.algorithm.value_loss_coef=0.5 \
    train.algorithm.gamma=0.99 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024,512,512,256]" \
    "train.policy.critic_hidden_dims=[1024,512,512,256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/0.3_3_4mlp" \