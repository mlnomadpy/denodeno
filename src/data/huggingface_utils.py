"""HuggingFace Hub integration utilities.

This module provides utilities for uploading datasets to HuggingFace Hub
and streaming data during training.
"""

import os
from typing import Optional, Iterator, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import jax.numpy as jnp


def upload_dataset_to_hub(
    local_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload dataset"
) -> str:
    """Upload a local dataset to HuggingFace Hub.
    
    Args:
        local_path: Path to local dataset directory.
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name').
        token: HuggingFace API token. If None, uses cached token.
        private: Whether to make the repository private.
        commit_message: Commit message for the upload.
        
    Returns:
        URL of the uploaded dataset.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Repository creation note: {e}")
    
    # Upload folder
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=commit_message
    )
    
    return f"https://huggingface.co/datasets/{repo_id}"


def load_dataset_from_hub(
    repo_id: str,
    split: str = "train",
    token: Optional[str] = None,
    streaming: bool = False
) -> Any:
    """Load a dataset from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID.
        split: Dataset split to load.
        token: HuggingFace API token.
        streaming: Whether to stream the dataset.
        
    Returns:
        HuggingFace Dataset object.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(
        repo_id,
        split=split,
        token=token,
        streaming=streaming
    )
    
    return dataset


def stream_dataset_from_hub(
    repo_id: str,
    split: str = "train",
    batch_size: int = 32,
    token: Optional[str] = None,
    image_column: str = "image",
    label_column: str = "label",
    image_size: Tuple[int, int] = (224, 224)
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Stream dataset from HuggingFace Hub with batching.
    
    Args:
        repo_id: HuggingFace repository ID.
        split: Dataset split to stream.
        batch_size: Batch size for streaming.
        token: HuggingFace API token.
        image_column: Name of the image column.
        label_column: Name of the label column.
        image_size: Size to resize images to.
        
    Yields:
        Batches of (images, labels) as JAX arrays.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(
        repo_id,
        split=split,
        token=token,
        streaming=True
    )
    
    batch_images = []
    batch_labels = []
    
    for example in dataset:
        # Process image
        if image_column in example:
            img = example[image_column]
            if hasattr(img, 'resize'):
                img = img.resize(image_size)
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Handle different image formats
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Convert to channels-first
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_array = (img_array - mean) / std
            
            batch_images.append(img_array)
        
        # Process label
        if label_column in example:
            batch_labels.append(example[label_column])
        
        # Yield batch when full
        if len(batch_images) >= batch_size:
            yield (
                jnp.stack(batch_images[:batch_size], axis=0),
                jnp.array(batch_labels[:batch_size])
            )
            batch_images = batch_images[batch_size:]
            batch_labels = batch_labels[batch_size:]
    
    # Yield remaining samples
    if batch_images:
        yield (
            jnp.stack(batch_images, axis=0),
            jnp.array(batch_labels)
        )


def stream_multimodal_from_hub(
    repo_id: str,
    split: str = "train",
    batch_size: int = 32,
    num_modalities: int = 5,
    token: Optional[str] = None,
    image_size: Tuple[int, int] = (224, 224)
) -> Iterator[Tuple[List[jnp.ndarray], List[jnp.ndarray]]]:
    """Stream multimodal dataset from HuggingFace Hub.
    
    Expects dataset with columns:
        - noisy_image, noiseless_image
        - noisy_signal, noiseless_signal
        - noise
    
    Args:
        repo_id: HuggingFace repository ID.
        split: Dataset split to stream.
        batch_size: Batch size.
        num_modalities: Number of modalities to return.
        token: HuggingFace API token.
        image_size: Size to resize images to.
        
    Yields:
        Batches of (inputs, targets) where each is a list of arrays.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    dataset = load_dataset(
        repo_id,
        split=split,
        token=token,
        streaming=True
    )
    
    modality_columns = [
        ('noisy_image', 'noiseless_image'),
        ('noisy_signal', 'noiseless_signal'),
        ('noiseless_image', 'noiseless_image'),
        ('noiseless_signal', 'noiseless_signal'),
        ('noise', 'noise')
    ][:num_modalities]
    
    batch_inputs = [[] for _ in range(num_modalities)]
    batch_targets = [[] for _ in range(num_modalities)]
    
    for example in dataset:
        for i, (input_col, target_col) in enumerate(modality_columns):
            # Process input
            if input_col in example:
                data = _process_modality_data(example[input_col], image_size)
                batch_inputs[i].append(data)
            
            # Process target
            if target_col in example:
                data = _process_modality_data(example[target_col], image_size)
                batch_targets[i].append(data)
        
        # Yield batch when full
        if len(batch_inputs[0]) >= batch_size:
            yield (
                [jnp.stack(batch_inputs[i][:batch_size], axis=0) for i in range(num_modalities)],
                [jnp.stack(batch_targets[i][:batch_size], axis=0) for i in range(num_modalities)]
            )
            for i in range(num_modalities):
                batch_inputs[i] = batch_inputs[i][batch_size:]
                batch_targets[i] = batch_targets[i][batch_size:]
    
    # Yield remaining
    if batch_inputs[0]:
        yield (
            [jnp.stack(batch_inputs[i], axis=0) for i in range(num_modalities)],
            [jnp.stack(batch_targets[i], axis=0) for i in range(num_modalities)]
        )


def _process_modality_data(
    data: Any,
    image_size: Tuple[int, int]
) -> np.ndarray:
    """Process modality data (image or signal).
    
    Args:
        data: Raw data (PIL Image or numpy array).
        image_size: Target image size.
        
    Returns:
        Processed numpy array with shape (C, H, W).
    """
    if hasattr(data, 'resize'):
        # It's a PIL Image
        img = data.resize(image_size)
        arr = np.array(img).astype(np.float32) / 255.0
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.transpose(arr, (2, 0, 1))
    else:
        # It's a numpy array (signal)
        arr = np.array(data)
        if arr.ndim == 1:
            # Reshape 1D signal to 2D
            target_length = image_size[0] * image_size[1]
            if len(arr) != target_length:
                x = np.linspace(0, len(arr) - 1, target_length)
                arr = np.interp(x, np.arange(len(arr)), arr.real)
            arr = arr.reshape(image_size[0], image_size[1])
        arr = np.stack([arr] * 3, axis=0).astype(np.float32)
        return arr


def create_dataset_card(
    repo_id: str,
    description: str,
    modulation_types: List[str],
    snr_range: Tuple[float, float],
    num_samples: int,
    token: Optional[str] = None
) -> None:
    """Create and upload a dataset card to HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID.
        description: Dataset description.
        modulation_types: List of modulation types in the dataset.
        snr_range: SNR range used for generation.
        num_samples: Total number of samples.
        token: HuggingFace API token.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
    
    card_content = f"""---
license: mit
task_categories:
  - image-classification
  - image-to-image
tags:
  - signal-processing
  - modulation-classification
  - denoising
size_categories:
  - 1K<n<10K
---

# {repo_id.split('/')[-1]}

{description}

## Dataset Description

This dataset contains constellation images for various modulation types,
generated for training denoising autoencoders and classification models.

### Modulation Types
{chr(10).join(f'- {mod}' for mod in modulation_types)}

### Generation Parameters
- **SNR Range**: {snr_range[0]} dB to {snr_range[1]} dB
- **Total Samples**: {num_samples}

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Stream for training
dataset = load_dataset("{repo_id}", streaming=True)
```

## License

MIT License
"""
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
