# MiniGrid Skills RL

Hierarchical RL for MiniGrid using skill-based action abstraction with Masked DQN.

## Quick Start

```bash
# Install
conda create -n minigrid_rl python=3.9 -y
conda activate minigrid_rl
pip install -r requirements.txt

# Train with skills (5 actions)
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-8x8-v0 --steps 50000 --save_dir results/skill_dqn

# Train with primitives (7 actions)
python -m experiments.doorkey.test_primitive_dqn \
    --env MiniGrid-DoorKey-8x8-v0 --steps 100000 --save_dir results/primitive_dqn

# Evaluate
python -m experiments.advanced_doorkey.core.skills_dqn_eval results/skill_dqn/agent
```

Add `--gpu` for CUDA.

## Skills

| Skill | Precondition |
|-------|--------------|
| GetKey | Key on grid |
| OpenDoor | (Locked + has key) OR (unlocked + closed) |
| CloseDoor | Door open |
| GoToGoal | Door open OR same room as goal |
| DropKey | Holding key |

## Core Files

| File | Purpose |
|------|---------|
| `experiments/advanced_doorkey/core/skills.py` | SkillEnv wrapper, skill definitions |
| `experiments/advanced_doorkey/core/masked_dqn.py` | DQN with action masking |
| `experiments/advanced_doorkey/core/skills_dqn_train.py` | Skill agent training |
| `experiments/doorkey/test_primitive_dqn.py` | Primitive agent training |
| `experiments/value_visualization.py` | Value function heatmaps |

## Value Visualization

Generates heatmaps every 100 episodes showing learned state values for:
- No key (agent in starting room)
- Has key (can open door)
- Door open (full access)

Saved to `{save_dir}/value_plots/`.

## Environments

- `MiniGrid-DoorKey-5x5-v0` - Fast iteration
- `MiniGrid-DoorKey-8x8-v0` - Recommended
- `MiniGrid-DoorKey-16x16-v0` - Large scale

## Output

```
results/{save_dir}/
├── agent/              # Checkpoint
├── value_plots/        # Heatmaps
├── training_curves.png
└── episode_rewards.npy
```
