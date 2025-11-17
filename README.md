# MiniGrid Skills RL

Reinforcement learning with hierarchical skills for MiniGrid environments using DQN and PPO.

## Overview

This repository implements skill-based RL agents that learn to solve MiniGrid tasks by selecting high-level skills (get_key, open_door, go_to_goal, etc.) rather than primitive actions.

## Key Features

- **Skill-based Environment Wrapper**: `SkillEnv` wrapper that converts MiniGrid into a skill-selection task
- **Masked DQN**: DQN agent with action masking to only select valid skills
- **Sparse Rewards**: Only rewards goal completion (prevents reward hacking)
- **Episode Limits**: Prevents infinite loops during training

## Setup

### Installation

```bash
# Create conda environment
conda create -n minigrid_rl python=3.9
conda activate minigrid_rl

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import gymnasium; import minigrid; import pfrl; print('All imports successful!')"
```

## Quick Start

### Train a DQN Agent

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-5x5-v0 \
    --steps 2000 \
    --save_dir results/dqn_agent \
    --seed 42
```

### Evaluate the Agent

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_agent/agent \
    --episodes 100 \
    --render
```

## Repository Structure

```
minigrid_skills_rl/
├── experiments/
│   ├── advanced_doorkey/
│   │   ├── core/
│   │   │   ├── skills.py              # SkillEnv wrapper
│   │   │   ├── masked_dqn.py          # Masked DQN agent
│   │   │   ├── skills_dqn_network.py  # Q-Network architecture
│   │   │   ├── skills_dqn_train.py    # Training script
│   │   │   └── skills_dqn_eval.py     # Evaluation script
│   │   └── DQN_GUIDE.md               # Quick reference guide
│   └── doorkey/                        # Additional experiments
├── results/                            # Training outputs
├── configs/                            # Configuration files
├── requirements.txt
└── README.md
```

## Skills

The agent can select from 5 skills:

1. **get_key**: Navigate to and pick up the key
2. **open_door**: Navigate to and open the door (requires key)
3. **close_door**: Navigate to and close the door
4. **go_to_goal**: Navigate to the goal location
5. **drop_key**: Drop the currently held key

Each skill uses A* pathfinding and primitive actions to achieve its objective.

## Training Details

- **Sparse Rewards**: Agent only receives +1.0 reward when reaching the goal
- **Action Masking**: Invalid skills are masked (e.g., can't open door without key)
- **Episode Limit**: Episodes truncate after 100 skill-level steps
- **Exploration**: Linear epsilon decay from 1.0 to 0.01

## Results

After training, you'll find:
- **Agent checkpoint**: `results/{save_dir}/agent/`
- **Training curves**: `results/{save_dir}/training_curves.png`
- **Evaluation metrics**: `results/{save_dir}/eval_results.npz`
- **Evaluation plots**: `results/{save_dir}/eval_plots.png`

## Common Issues

### Agent loops between get_key and drop_key
- This indicates training with the old reward structure
- Retrain with the current code (sparse rewards only)

### No movement during evaluation
- Check that skills are completing: look for "Skill completed!" messages
- Verify the agent checkpoint exists and loaded correctly

## Citation

This code is based on research in hierarchical reinforcement learning and portable options.

## License

[Add your license here]
