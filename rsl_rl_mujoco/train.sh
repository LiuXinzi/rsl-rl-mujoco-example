#!/usr/bin/env bash


python train.py \
    train.algorithm.value_loss_coef=0.7 \
    train.algorithm.gamma=0.99 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024, 512, 256]" \
    "train.policy.critic_hidden_dims=[1024, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/1" \

python train.py \
    train.algorithm.value_loss_coef=0.3 \
    train.algorithm.gamma=0.99 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024, 512, 256]" \
    "train.policy.critic_hidden_dims=[1024, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/2" \

python train.py \
    train.algorithm.value_loss_coef=0.5 \
    train.algorithm.gamma=0.95 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024, 512, 256]" \
    "train.policy.critic_hidden_dims=[1024, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/3" \

python train.py \
    train.algorithm.value_loss_coef=0.5 \
    train.algorithm.gamma=0.999 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024, 512, 256]" \
    "train.policy.critic_hidden_dims=[1024, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/4" \

python train.py \
    train.algorithm.value_loss_coef=0.5 \
    train.algorithm.gamma=0.99 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[512, 512, 256]" \
    "train.policy.critic_hidden_dims=[512, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/5" \

python train.py \
    train.algorithm.value_loss_coef=0.5 \
    train.algorithm.gamma=0.99 \
    train.algorithm.learning_rate=0.0001 \
    "train.policy.actor_hidden_dims=[1024, 512, 512, 256]" \
    "train.policy.critic_hidden_dims=[1024, 512, 512, 256]" \
    train.policy.init_noise_std=0.1 \
    "log_dir= ./logs/6" \