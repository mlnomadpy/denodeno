#!/bin/bash
# Fine-tune DenoMAE for classification

python -m src.evaluation.finetune \
    --pretrained_model_path "models/denomae_5.pkl" \
    --data_path "./data/labeled/0_dB" \
    --save_model_path "models/finetuned_denomae.pkl" \
    --img_size 224 \
    --patch_size 16 \
    --in_chans 3 \
    --embed_dim 768 \
    --encoder_depth 12 \
    --decoder_depth 4 \
    --num_heads 12 \
    --num_modality 1 \
    --num_classes 10 \
    --batch_size 32 \
    --num_epochs 150 \
    --learning_rate 1e-4 \
    --freeze_encoder \
    --wandb_project "denomae-finetune" \
    --confusion_matrix_path "results/confusion_matrix.png" \
    --use_tpu \
    "$@"
