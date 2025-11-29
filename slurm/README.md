# SLURM Scripts for Oscar (Brown HPC)

## Setup (First Time)

```bash
# On Oscar, create conda environment
module load anaconda/2022.05
conda create -n minigrid_rl python=3.9 -y
conda activate minigrid_rl
pip install -r requirements.txt

# Create log directory
mkdir -p slurm/logs
```

## Submit Jobs

### Skill-based DQN (5 skills)

```bash
# Default: 16x16, 100k steps
sbatch slurm/train_skills.sh

# Custom: environment size, steps, seed
sbatch slurm/train_skills.sh 8x8 50000 42
sbatch slurm/train_skills.sh 16x16 200000 123
```

### Primitive DQN (7 actions)

```bash
# Default: 8x8, 200k steps
sbatch slurm/train_primitive.sh

# Custom
sbatch slurm/train_primitive.sh 16x16 500000 42
```

## Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f slurm/logs/minigrid_skills_<JOB_ID>.out

# Check GPU partition status
allq gpu

# Cancel a job
scancel <JOB_ID>
```

## Output

Results saved to:
```
results/oscar_skills_<ENV>_<JOB_ID>/
├── agent/              # Model checkpoint
├── value_plots/        # Value function heatmaps
├── training_curves.png
├── episode_rewards.npy
└── episode_lengths.npy
```

Logs saved to:
```
slurm/logs/
├── minigrid_skills_<JOB_ID>.out   # stdout
└── minigrid_skills_<JOB_ID>.err   # stderr
```

## GPU Selection

Scripts use `--constraint=ampere` for modern GPUs (A5000, A5500, A6000, 3090).

Available GPU types on Oscar:
- `ampere` - A5000, A5500, A6000, GeForce 3090 (24-48GB)
- `v100` - Tesla V100 (32GB, NVLink)
- `quadrortx` - Quadro RTX (24GB)

To use a specific GPU type, modify the `--constraint` line in the script.

## Troubleshooting

**Job pending too long?**
- Try removing `--constraint=ampere` for any available GPU
- Use `gpu-debug` partition for quick tests: `--partition=gpu-debug --time=00:30:00`

**Out of memory?**
- Increase `--mem` (up to 1028G on some nodes)
- Use `--obs_size 64` for smaller observations

**CUDA not available?**
- Check `module load cuda/11.8.0` is in script
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
