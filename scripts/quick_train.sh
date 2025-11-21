#!/bin/bash
# Quick training script for common scenarios

set -e

echo "========================================"
echo "AUV Obstacle Avoidance - Quick Training"
echo "========================================"

# Parse arguments
ALGO=${1:-ppo}
CONFIG=${2:-with_obstacle}
TIMESTEPS=${3:-1000000}
N_ENVS=${4:-4}
DEVICE=${5:-auto}

echo ""
echo "Configuration:"
echo "  Algorithm: $ALGO"
echo "  Environment: $CONFIG"
echo "  Timesteps: $TIMESTEPS"
echo "  Parallel Envs: $N_ENVS"
echo "  Device: $DEVICE"
echo "========================================"
echo ""

# Run training
python train_agent.py \
    --algo $ALGO \
    --config $CONFIG \
    --timesteps $TIMESTEPS \
    --n-envs $N_ENVS \
    --device $DEVICE \
    --eval-freq 10000 \
    --n-eval-episodes 5 \
    --save-freq 50000 \
    --verbose 1

echo ""
echo "========================================"
echo "Training completed!"
echo "Check models/ directory for saved models"
echo "View training progress: tensorboard --logdir tensorboard/"
echo "========================================"

