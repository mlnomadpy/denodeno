<p align="center">
  <img src="denoMAE.jpeg" alt="DenoMAE Logo" width="300"/>
</p>

# DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals

DenoMAE is a novel multimodal autoencoder framework that extends masked autoencoding for denoising and classifying modulation signals. It achieves state-of-the-art accuracy in automatic modulation classification with significantly reduced data requirements, while exhibiting robust performance across varying Signal-to-Noise Ratios (SNRs).

**This version has been migrated from PyTorch to JAX/Flax NNX with:**
- Pure data parallel training support for TPU v5-8 (8 chips)
- HuggingFace Hub integration for data upload and streaming
- Weights & Biases (WandB) integration for experiment tracking
- Modular test-driven architecture

[Paper on IEEE Xplore](https://ieeexplore.ieee.org/document/11005616)

## Project Overview

DenoMAE consists of two main phases:
1. **Pretraining**: Training a masked autoencoder to reconstruct signals from partially masked inputs
2. **Fine-tuning**: Using the pretrained model for classification tasks on constellation signals

The model is based on Vision Transformer (ViT) architecture, adapted for signal processing applications.

## Directory Structure

```
denodeno/
├── src/                         # Main source code
│   ├── denomae/                 # Core DenoMAE models
│   │   ├── encoder_decoder.py   # Transformer encoder/decoder
│   │   ├── model.py             # DenoMAE and FineTunedDenoMAE
│   │   └── mesh_utils.py        # TPU mesh utilities
│   ├── data/                    # Data generation and loading
│   │   ├── sample_generation.py # JAX-based sample generation
│   │   ├── datagen.py           # Data loading utilities
│   │   └── huggingface_utils.py # HuggingFace integration
│   ├── training/                # Training utilities
│   │   ├── trainer.py           # Training loop and utilities
│   │   ├── train_denomae.py     # Main training script
│   │   └── wandb_utils.py       # WandB integration
│   └── evaluation/              # Evaluation utilities
│       ├── evaluator.py         # Evaluation functions
│       └── finetune.py          # Fine-tuning script
├── tests/                       # Unit tests
│   ├── test_encoder_decoder.py
│   ├── test_model.py
│   ├── test_sample_generation.py
│   ├── test_mesh_utils.py
│   └── test_trainer.py
├── configs/                     # Configuration files
│   └── default_config.py
├── scripts/                     # Shell scripts
│   ├── train.sh                 # Training script
│   ├── finetune.sh              # Fine-tuning script
│   ├── generate_samples.sh      # Data generation script
│   └── run_tests.sh             # Test runner
├── models/                      # Saved model weights
├── sample_generation/           # Legacy PyTorch sample generation
├── requirements.txt             # Python dependencies
└── README.md
```

## Requirements

- Python 3.10+
- JAX with TPU support (for TPU v5-8)
- Flax NNX >= 0.8.0

```bash
pip install -r requirements.txt
```

### TPU Setup

For TPU v5-8 training, ensure you have the correct JAX version:

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

### 1. Generate Training Data

```bash
chmod +x scripts/generate_samples.sh
./scripts/generate_samples.sh
```

### 2. Train DenoMAE

```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

Or directly with Python:

```bash
python -m src.training.train_denomae \
    --train_path ./data/unlabeled/train/ \
    --test_path ./data/unlabeled/test/ \
    --num_modality 5 \
    --num_epochs 100 \
    --use_tpu
```

### 3. Fine-tune for Classification

```bash
chmod +x scripts/finetune.sh
./scripts/finetune.sh
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## Features

### TPU v5-8 Data Parallelism

The framework supports pure data parallel training on TPU v5-8 (8 chips):

```python
from src.denomae.mesh_utils import create_device_mesh, DataParallelTrainer

# Create mesh for TPU v5-8
mesh = create_device_mesh(mesh_shape=(8,), axis_names=('data',))
trainer = DataParallelTrainer(mesh)

# Shard batches across devices
sharded_batch = trainer.shard_batch(batch)
```

### HuggingFace Integration

Upload and stream data from HuggingFace Hub:

```python
from src.data.huggingface_utils import upload_to_huggingface, HuggingFaceDataLoader

# Upload data
upload_to_huggingface(
    data_dir='./data/unlabeled/train',
    repo_id='your-username/denomae-data',
    token='your-hf-token'
)

# Stream data for training
loader = HuggingFaceDataLoader(
    repo_id='your-username/denomae-data',
    split='train',
    batch_size=64
)
```

### WandB Integration

Track experiments with Weights & Biases:

```python
from src.training.wandb_utils import WandBLogger

with WandBLogger(
    project='denomae',
    name='experiment-1',
    config={'learning_rate': 1e-4}
) as logger:
    # Training loop
    logger.log({'loss': 0.5, 'accuracy': 0.95})
    logger.log_images({'reconstruction': recon_image})
```

## Model Architecture

DenoMAE uses a Vision Transformer (ViT) architecture with:
- Patch embedding layer converting input signals into tokens
- Transformer encoder with self-attention layers (using `nnx.List`)
- Transformer decoder for reconstruction (using `nnx.List`)
- Modality-specific heads for multi-modal learning

Key implementation details:
- Uses Flax NNX `nnx.List` for module collections (best practice)
- Supports arbitrary number of modalities
- Configurable masking ratio for self-supervised learning

## Sample Generation

Generate constellation images for various modulation types:

```python
from src.data.sample_generation import generate_constellation_images

generate_constellation_images(
    mod_type='QPSK',
    samples_per_image=1024,
    image_num=100,
    image_size=(224, 224),
    set_types=['noisyImg', 'noiseLessImg'],
    set_path='./data/train',
    snr_range=(-10, 10)
)
```

Supported modulation types:
- OOK, BPSK, QPSK, 8PSK
- 4ASK, 8ASK
- 4PAM, 16PAM
- OQPSK, DQPSK
- CPFSK, GFSK, 4FSK
- 16QAM
- GMSK

## API Reference

### DenoMAE Model

```python
from src.denomae import DenoMAE
from flax import nnx

rngs = nnx.Rngs(jax.random.PRNGKey(0))

model = DenoMAE(
    num_modalities=5,
    img_size=224,
    patch_size=16,
    in_chans=3,
    mask_ratio=0.75,
    embed_dim=768,
    encoder_depth=12,
    decoder_depth=4,
    num_heads=12,
    rngs=rngs
)

# Forward pass
reconstructions, masks = model(inputs, key)
```

### Fine-tuned Model

```python
from src.denomae import FineTunedDenoMAE

finetuned = FineTunedDenoMAE(
    backbone=pretrained_model,
    num_classes=10,
    freeze_encoder=True,
    rngs=rngs
)

logits = finetuned(images)
```

## Testing

Run the test suite:

```bash
# All tests
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_model.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{denomae2024,
  title={DenoMAE: A Multimodal Autoencoder for Denoising Modulation Signals},
  author={...},
  journal={IEEE},
  year={2024}
}
```

## License

MIT License
