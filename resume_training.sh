#!/bin/bash

# Resume training script with CUDA error recovery
# This script will automatically restart training from the last checkpoint if CUDA errors occur

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Change to project directory
cd /home/sunny/coding/COMP4471/Progressive-GAN-pytorch

# Find the most recent trial directory for ffhq_production_v2
TRIAL_DIR=$(ls -td trial_ffhq_production_v2* 2>/dev/null | head -1)
if [ -z "$TRIAL_DIR" ]; then
    CHECKPOINT_DIR=""
else
    CHECKPOINT_DIR="${TRIAL_DIR}/checkpoint"
fi

# Activate conda environment
source $HOME/miniconda3/bin/activate
conda activate ai

# Start training with automatic restart on failure
while true; do
    # Find latest checkpoint
    if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
        LATEST_CHECKPOINT=""
    else
        LATEST_CHECKPOINT=$(ls -t ${CHECKPOINT_DIR}/*_g.model 2>/dev/null | head -1)
    fi
    
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "==================================="
        echo "No checkpoint found, starting fresh training at $(date)"
        RESUME_ITER=0
        INIT_STEP=1
    else
        # Extract iteration number from checkpoint filename
        RESUME_ITER=$(basename "$LATEST_CHECKPOINT" | sed 's/^0*//' | sed 's/_g.model//')
        echo "==================================="
        echo "Found checkpoint at iteration: $RESUME_ITER"
        
        # Calculate which step we should be at
        if [ $RESUME_ITER -lt 100000 ]; then
            INIT_STEP=1
        elif [ $RESUME_ITER -lt 200000 ]; then
            INIT_STEP=2
        elif [ $RESUME_ITER -lt 300000 ]; then
            INIT_STEP=3
        elif [ $RESUME_ITER -lt 400000 ]; then
            INIT_STEP=4
        elif [ $RESUME_ITER -lt 500000 ]; then
            INIT_STEP=5
        else
            INIT_STEP=6
        fi
    fi
    
    echo "Starting/Resuming training from iteration $RESUME_ITER at step $INIT_STEP ($(date))"
    echo "==================================="
    
    # Build command with checkpoint arguments if resuming
    if [ $RESUME_ITER -gt 0 ]; then
        CHECKPOINT_BASE=$(dirname "$LATEST_CHECKPOINT")
        CHECKPOINT_G="${CHECKPOINT_BASE}/$(printf "%06d" $RESUME_ITER)_g.model"
        CHECKPOINT_D="${CHECKPOINT_BASE}/$(printf "%06d" $RESUME_ITER)_d.model"
        
        python train.py \
            --path /home/sunny/coding/COMP4471/ffhq-dataset/images1024x1024 \
            --trial_name ffhq_production_v2 \
            --gpu_id 0 \
            --lr 0.001 \
            --z_dim 512 \
            --channel 512 \
            --batch_size 12 \
            --n_critic 1 \
            --init_step $INIT_STEP \
            --total_iter 600000 \
            --pixel_norm \
            --tanh \
            --checkpoint_g "$CHECKPOINT_G" \
            --checkpoint_d "$CHECKPOINT_D" \
            --resume_iter $RESUME_ITER
    else
        python train.py \
            --path /home/sunny/coding/COMP4471/ffhq-dataset/images1024x1024 \
            --trial_name ffhq_production_v2 \
            --gpu_id 0 \
            --lr 0.001 \
            --z_dim 512 \
            --channel 512 \
            --batch_size 12 \
            --n_critic 1 \
            --init_step $INIT_STEP \
            --total_iter 600000 \
            --pixel_norm \
            --tanh
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "==================================="
        echo "Training completed successfully at $(date)!"
        echo "==================================="
        break
    else
        echo "==================================="
        echo "Training crashed with exit code $EXIT_CODE at $(date)"
        echo "Waiting 30 seconds before restart..."
        echo "==================================="
        sleep 30
    fi
done
