#!/bin/bash
#SBATCH --job-name=complex_doorkey
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@brown.edu

# ============================================================================
# Complex DoorKey Training Script
# ============================================================================
# Task: Agent must drop key in first room, close door, then reach goal
# This is a harder task requiring ~8 skill sequence with delayed reward
#
# Usage:
#   sbatch slurm/train_complex_doorkey.sh                    # Default settings
#   sbatch slurm/train_complex_doorkey.sh 200000 42          # 200k steps, seed 42
#   sbatch slurm/train_complex_doorkey.sh 500000 123         # 500k steps, seed 123
# ============================================================================

# Parse arguments with defaults (more steps needed for this hard task)
STEPS="${1:-300000}"
SEED="${2:-42}"

# Fixed settings for complex_doorkey
ENV_SIZE="16x16"
REWARD_MODE="complex_doorkey"
ENV_NAME="MiniGrid-DoorKey-${ENV_SIZE}-v0"
SAVE_DIR="results/oscar_${REWARD_MODE}_${SLURM_JOB_ID}"

# Print job info
echo "=============================================="
echo "Complex DoorKey Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Reward Mode: $REWARD_MODE"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo "Save Dir: $SAVE_DIR"
echo "=============================================="
echo ""
echo "Task: Drop key in first room + close door + reach goal"
echo "Expected sequence: GetKey -> OpenDoor -> navigate -> DropKey -> CloseDoor -> GoToGoal"
echo "=============================================="

# Create log directory
mkdir -p slurm/logs

# Load modules
module purge
module load anaconda/2023.09-0-7nso27y
module load cuda/11.8.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid_rl

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Navigate to project directory
cd /users/lzheng35/Desktop/brown/irl-lab/minigrid_skills_rl

# Run training
echo "Starting training..."
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env "$ENV_NAME" \
    --steps "$STEPS" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    --use_gpu \
    --reward_mode "$REWARD_MODE" \
    --obs_size 84

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Training completed successfully!"
    echo "Results saved to: $SAVE_DIR"
    echo "Finished: $(date)"
    echo "=============================================="
else
    echo ""
    echo "=============================================="
    echo "Training FAILED with exit code: $EXIT_CODE"
    echo "Check error log: slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
    echo "=============================================="
fi

exit $EXIT_CODE
