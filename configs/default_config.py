"""Default configuration for DenoMAE training."""

# Model configuration
MODEL_CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'in_chans': 3,
    'mask_ratio': 0.75,
    'embed_dim': 768,
    'encoder_depth': 12,
    'decoder_depth': 4,
    'num_heads': 12,
    'num_modalities': 5,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'log_interval': 50,
}

# Data configuration
DATA_CONFIG = {
    'train_path': './data/unlabeled/train/',
    'test_path': './data/unlabeled/test/',
    'image_size': (224, 224),
    'target_length': 50176,
    'normalize': True,
}

# TPU configuration for v5-8
TPU_CONFIG = {
    'mesh_shape': (8,),
    'axis_names': ('data',),
    'use_data_parallel': True,
}

# WandB configuration
WANDB_CONFIG = {
    'project': 'denomae',
    'entity': None,  # Set to your wandb username or team
    'tags': ['denomae', 'jax', 'flax-nnx'],
}

# HuggingFace configuration
HF_CONFIG = {
    'repo_id': None,  # Set to your repo ID for streaming
    'private': False,
}

# Sample generation configuration
SAMPLE_GEN_CONFIG = {
    'samples_per_image': 1024,
    'image_size': (224, 224),
    'image_num': 100,
    'mod_types': ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK'],
    'set_types': ['noiseLessImg', 'noisyImg', 'noiselessSignal', 'noise', 'noisySignal'],
    'snr_range': (-10, 10),
}

# Fine-tuning configuration
FINETUNE_CONFIG = {
    'num_classes': 10,
    'freeze_encoder': True,
    'num_epochs': 150,
    'learning_rate': 1e-4,
    'batch_size': 32,
}
