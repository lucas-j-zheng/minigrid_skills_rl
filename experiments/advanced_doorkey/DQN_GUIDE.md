# DQN Skills Training & Evaluation

## Training
```bash
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_train \
    --env MiniGrid-DoorKey-5x5-v0 \
    --steps 50000 \
    --save_dir results/dqn_agent \
    --seed 42 \
    [--use_gpu]
```

## Evaluation
```bash
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_eval \
    results/dqn_agent/agent \
    --episodes 100 \
    [--render] \
    [--seed 42]
```
\