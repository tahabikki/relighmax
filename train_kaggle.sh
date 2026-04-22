#!/bin/bash
# Training/Fine-tuning script for Kaggle
# Usage: bash train_kaggle.sh

echo "=========================================="
echo "RetinexNet Fine-tuning for Kaggle"
echo "=========================================="

# Install dependencies
pip install -r requirements.txt -q

# Run fine-tuning
python finetune.py \
    --phase=train \
    --epoch=50 \
    --batch_size=8 \
    --patch_size=96 \
    --start_lr=0.0001 \
    --eval_every_epoch=10

echo "=========================================="
echo "Fine-tuning complete!"
echo "Checkpoints: ./checkpoint/"
echo "=========================================="