#!/usr/bin/env python3
import os
import time
import yaml
import torch
import argparse
import gymnasium as gym
from pathlib import Path
from rsl_rl.modules import ActorCritic
from rsl_rl_mujoco.env_wrapper import GymMujocoWrapper

from pathlib import Path

class PolicyVisualizer:
    def __init__(self, cfg_path):
        self.load_config(cfg_path)
        self.setup_environment()
        self.load_policy()

    def load_config(self, cfg_path):
        """Load visualization configuration"""
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Set default device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self):
        """Create and configure the environment"""
        env_kwargs = {
            "render_mode": self.cfg["env"]["render_mode"],
        }
        
        # Add video recording if enabled
        if self.cfg["visualization"]["record_video"]:
            os.makedirs(self.cfg["visualization"]["video_dir"], exist_ok=True)
            env_kwargs.update({
                "render_mode": "rgb_array",
                "width": self.cfg["env"]["width"],
                "height": self.cfg["env"]["height"],
            })
        
        # Create base environment
        self.env = gym.make(
            self.cfg["env"]["id"],
            **env_kwargs
        )
        
        # Wrap for RSL-RL compatibility
        self.env = GymMujocoWrapper(
            self.env,
            device=self.device,
            is_finite_horizon=False
        )

        # Optional: Add video recorder wrapper
        if self.cfg["visualization"]["record_video"]:
            from gymnasium.wrappers import RecordVideo
            self.env = RecordVideo(
                self.env,
                self.cfg["visualization"]["video_dir"],
                episode_trigger=lambda x: True,
                name_prefix="policy_visualization"
            )

    def load_policy(self):
        """Load trained policy network"""
        self.policy = ActorCritic(
            num_actions=self.env.num_actions,
            num_actor_obs=self.env.num_obs,
            num_critic_obs=self.env.num_obs,
            hidden_dims=self.cfg["policy"]["hidden_dims"]
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.cfg["policy"]["checkpoint"])
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy.eval()
        print(f"Loaded policy from {self.cfg['policy']['checkpoint']}")

    def run(self):
        """Run visualization loop"""
        print(f"Starting visualization for {self.cfg['visualization']['num_episodes']} episodes...")
        
        for episode in range(self.cfg["visualization"]["num_episodes"]):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    actions = self.policy.act_inference(obs)
                
                obs, reward, done, _ = self.env.step(actions)
                episode_reward += reward.item()
                
                # Control playback speed
                time.sleep(1.0 / (self.env.max_episode_length * self.cfg["visualization"]["speedup"]))
            
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
        
        self.env.close()

def main():
    parser = argparse.ArgumentParser(description="RSL-RL Policy Visualizer")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "visualize.yaml"),
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    visualizer = PolicyVisualizer(args.config)
    visualizer.run()

if __name__ == "__main__":
    main()