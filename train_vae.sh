#!/bin/bash

# IMPROVED VAE Training - No Teacher Forcing
# Optimized parameters for better generation capability

python3 train_vae.py \
    --epochs 350 \
    --batch-size 64 \
    --lr 2e-4 \
    --lambda-behavior 0.0 \
    --save-freq 10 \
    --resume checkpoints/latest_model.pt \
    --checkpoint-dir checkpoints/vae3