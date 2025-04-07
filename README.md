# RSL-RL with MuJoCo and Gymnasium Example

This repository provides an example of how to use RSL-RL with MuJoCo environments from Gymnasium.

## Installation

1. Install this package:
   ```bash
   pip install -e .
   ```

## Usage

Train the agent:
```bash
python -m rsl_rl_mujoco.train
```

## Configuration

Modify `configs/default.yaml` to change:
- Environment settings
- Training parameters
- Algorithm hyperparameters

## Supported Environments

Any MuJoCo environment from Gymnasium should work, such as:
- HalfCheetah-v4
- Ant-v4
- Humanoid-v4
- etc.
