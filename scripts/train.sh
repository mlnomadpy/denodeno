#!/bin/bash
# Train DenoMAE model using JAX/Flax

# Default configuration for TPU v5-8 (8 chips)
python -m src.training.train_denomae \
    --train_path "./data/unlabeled/train/" \
    --test_path "./data/unlabeled/test/" \
    --batch_size 64 \
    --image_size 224 224 \
    --patch_size 16 \
    --in_chans 3 \
    --mask_ratio 0.75 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --decoder_depth 4 \
    --num_heads 12 \
    --num_modality 5 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 0.05 \
    --model_dir "models" \
    --model_name "denomae" \
    --wandb_project "denomae" \
    --log_interval 50 \
    --use_tpu \
    --tpu_mesh_shape 8 \
    "$@"
