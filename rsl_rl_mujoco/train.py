import random
import numpy as np
import torch
import yaml
import gymnasium as gym
from RL.runners import OnPolicyRunner
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from rsl_rl_mujoco.env_wrapper import SB3RslVecEnv
from pathlib import Path
cfg_path: str = str(Path(__file__).resolve().parent / "configs" / "default.yaml")
with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
SEED = cfg["seed"]
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

gym.register(
    id='MFG_MS_V7',
    entry_point='Env.MFG_Musculoskelet_V7.mfgenv.mfg_env:MFG_Musculoskelet_V7',
    max_episode_steps=1000
)

if __name__ == "__main__":
    # Load configuration
    

    # Create environment
    env_id = cfg["env"]["id"]
    num_envs = cfg["env"].get("num_envs", 4)
    # import ipdb;ipdb.set_trace()
    envs = make_vec_env('MFG_MS_V7', n_envs=num_envs,seed=SEED,vec_env_cls=SubprocVecEnv)
    
    # envs = [gym.make(env_id) for _ in range(num_envs)]
    env = SB3RslVecEnv(
        envs,
        clip_actions=cfg["env"].get("clip_actions",None),
        is_finite_horizon=cfg["env"].get("is_finite_horizon", True),
        device=cfg.get("device", "cpu"),
        seed=SEED
    )

    # Initialize runner
    runner = OnPolicyRunner(
        env,
        cfg["train"],
        log_dir=cfg.get("log_dir", "./logs"),
        device=cfg.get("device", "cpu"),
    )

    # Train
    runner.learn(num_learning_iterations=cfg["train"]["num_learning_iterations"])