#!/bin/bash

# IMPROVED VAE Training - No Teacher Forcing
# Optimized parameters for better generation capability

python3 train_vae.py \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-5 \
    --beta 0.5 \
    --lambda-behavior 0.0