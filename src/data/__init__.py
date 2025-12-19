"""Data generation and loading utilities for DenoMAE.

This module provides JAX-based signal generation, image generation,
and data loading utilities for training and evaluation.
"""

from src.data.sample_generation import (
    awgn,
    gmsk_modulate,
    generate_image,
    generate_constellation_images,
    MOD_TYPES,
)

from src.data.datagen import (
    DenoMAEDataset,
    ImageFolderDataset,
    create_dataloader,
    create_classification_loader,
)

from src.data.huggingface_utils import (
    upload_dataset_to_hub,
    load_dataset_from_hub,
    stream_dataset_from_hub,
)

__all__ = [
    # Sample generation
    'awgn',
    'gmsk_modulate',
    'generate_image',
    'generate_constellation_images',
    'MOD_TYPES',
    # Data loading
    'DenoMAEDataset',
    'ImageFolderDataset',
    'create_dataloader',
    'create_classification_loader',
    # HuggingFace utilities
    'upload_dataset_to_hub',
    'load_dataset_from_hub',
    'stream_dataset_from_hub',
]
