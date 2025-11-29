#!/bin/bash
#SBATCH --job-name=minigrid_skills
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$USER@brown.edu

# ============================================================================
# MiniGrid Skills DQN Training Script for Oscar (Brown HPC)
# ============================================================================
# Usage:
#   sbatch slurm/train_skills.sh                    # Default 16x16
#   sbatch slurm/train_skills.sh 16x16 200000       # 16x16 with 200k steps
#   sbatch slurm/train_skills.sh 5x5 50000 123      # 5x5, 50k steps, seed 123
# ============================================================================

# Parse arguments with defaults
ENV_SIZE="${1:-16x16}"
STEPS="${2:-100000}"
SEED="${3:-42}"

# Derived variables
ENV_NAME="MiniGrid-DoorKey-${ENV_SIZE}-v0"
SAVE_DIR="results/oscar_skills_${ENV_SIZE}_${SLURM_JOB_ID}"

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================================="
echo "Environment: $ENV_NAME"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo "Save Dir: $SAVE_DIR"
echo "=============================================="

# Create log directory
mkdir -p slurm/logs

# Load modules
module purge
module load anaconda/2022.05
module load cuda/11.8.0

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate minigrid_rl

# Print environment info
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""

# Set observation size for larger environments
OBS_SIZE_ARG=""
if [[ "$ENV_SIZE" == "16x16" ]]; then
    OBS_SIZE_ARG="--obs_size 84"
    echo "Using resized observations (84x84) for 16x16 environment"
fi

# Navigate to project directory
cd $HOME/minigrid_skills_rl

# Run training
echo "Starting training..."
python -m experiments.advanced_doorkey.core.skills_dqn_train \
    --env "$ENV_NAME" \
    --steps "$STEPS" \
    --save_dir "$SAVE_DIR" \
    --seed "$SEED" \
    --use_gpu \
    $OBS_SIZE_ARG

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
