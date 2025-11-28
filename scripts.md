# Training Scripts

Run in `minigrid_rl` conda environment.

## 5x5 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-5x5-v0 \
    --steps 20000 \
    --save_dir results/dqn_5x5 \
    --seed 42
```

## 6x6 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-6x6-v0 \
    --steps 30000 \
    --save_dir results/dqn_6x6 \
    --seed 42
```

## 8x8 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-8x8-v0 \
    --steps 50000 \
    --save_dir results/dqn_8x8 \
    --seed 42
```

## 16x16 DoorKey

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-16x16-v0 \
    --steps 100000 \
    --save_dir results/dqn_16x16 \
    --seed 42
```

## With GPU

Add `--use_gpu` flag:

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-8x8-v0 \
    --steps 50000 \
    --save_dir results/dqn_8x8_gpu \
    --seed 42 \
    --use_gpu
```

## Evaluation

### 5x5

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_5x5/agent \
    --env MiniGrid-DoorKey-5x5-v0 \
    --episodes 100 \
    --seed 42 \
    --render
```

### 6x6

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_6x6/agent \
    --env MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --seed 42 \
    --render
```

### 8x8

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_8x8/agent \
    --env MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --seed 42 \
    --render
```

### 16x16

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_16x16/agent \
    --env MiniGrid-DoorKey-16x16-v0 \
    --episodes 100 \
    --seed 42 \
    --render
```

### Without rendering (faster)

```bash
python -m experiments.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_8x8/agent \
    --env MiniGrid-DoorKey-8x8-v0 \
    --episodes 100 \
    --seed 42
```

### Analyze skill sequences

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
