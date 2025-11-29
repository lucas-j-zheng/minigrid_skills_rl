# Training Scripts

Run in `minigrid_rl` conda environment.

## Observation Mode

Training now uses **full top-down RGB view** (not partial 7x7 egocentric view).

| Grid Size | Observation Size |
|-----------|------------------|
| 5x5       | 40x40x3          |
| 8x8       | 64x64x3          |
| 16x16     | 128x128x3        |

Use `--obs_size 84` to resize large observations for efficiency.

## Debug: Check observations

```bash
# Visualize observation for any environment
python debug_observation.py --env MiniGrid-DoorKey-8x8-v0 --full

# Compare all sizes
python debug_observation.py --compare
```

## Skill-based DQN (5 skills)

### 5x5 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-5x5-v0 \
    --steps 20000 \
    --save_dir results/dqn_5x5 \
    --seed 42
```

### 8x8 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-8x8-v0 \
    --steps 50000 \
    --save_dir results/dqn_8x8 \
    --seed 42
```

### 16x16 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-16x16-v0 \
    --steps 100000 \
    --save_dir results/dqn_16x16 \
    --seed 42 \
    --obs_size 84
```

### With GPU

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-8x8-v0 \
    --steps 50000 \
    --save_dir results/dqn_8x8_gpu \
    --seed 42 \
    --use_gpu
```

## Primitive DQN (7 actions)

### 8x8 DoorKey

```bash
python -m experiments.doorkey.test_primitive_dqn \
    --env MiniGrid-DoorKey-8x8-v0 \
    --steps 100000 \
    --save_dir results/primitive_8x8 \
    --seed 42
```

### 16x16 DoorKey (with resizing)

```bash
python -m experiments.doorkey.test_primitive_dqn \
    --env MiniGrid-DoorKey-16x16-v0 \
    --steps 200000 \
    --save_dir results/primitive_16x16 \
    --seed 42 \
    --obs_size 84
```

## Evaluation

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_8x8/agent \
    --env MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --seed 42 \
    --render
```

Without rendering (faster):

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_8x8/agent \
    --env MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --seed 42
```

Analyze skill sequences:

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_8x8/agent \
    --env MiniGrid-DoorKey-8x8-v0 \
    --seed 42 \
    --analyze_sequences
```

## Output

Each training run produces:
- `results/{save_dir}/agent/` - Model checkpoint
- `results/{save_dir}/training_curves.png` - Loss/reward plots
- `results/{save_dir}/value_plots/` - Value function heatmaps (every 100 episodes)
- `results/{save_dir}/episode_rewards.npy` - Raw metrics
