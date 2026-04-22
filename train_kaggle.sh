# RetinexNet Training Script for Kaggle
# Usage: python main.py --phase=train [parameters]

python main.py \
    --phase=train \
    --epoch=100 \
    --batch_size=8 \
    --patch_size=96 \
    --start_lr=0.001 \
    --eval_every_epoch=10 \
    --checkpoint_dir=./checkpoint \
    --sample_dir=./sample