"""Data loading utilities for DenoMAE training and evaluation.

This module provides dataset classes and data loading utilities compatible
with JAX training pipelines.
"""

import os
from typing import Tuple, List, Iterator, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image
import jax.numpy as jnp


class DenoMAEDataset:
    """Dataset for DenoMAE multimodal training.
    
    Loads images and signals for denoising autoencoder training.
    
    Args:
        num_modalities: Number of modalities to return.
        noisy_image_path: Directory containing noisy images.
        noiseless_img_path: Directory containing noiseless images.
        noisy_signal_path: Directory containing noisy signals.
        noiseless_signal_path: Directory containing noiseless signals.
        noise_path: Directory containing noise arrays.
        image_size: Size to resize images to.
        target_length: Target length for signal arrays.
    """
    
    def __init__(
        self,
        num_modalities: int,
        noisy_image_path: str,
        noiseless_img_path: str,
        noisy_signal_path: str,
        noiseless_signal_path: str,
        noise_path: str,
        image_size: Tuple[int, int] = (224, 224),
        target_length: int = 50176
    ):
        self.num_modalities = num_modalities
        self.image_size = image_size
        self.target_length = target_length
        
        # Use pathlib for path handling
        self.paths = {
            'noisy_image': Path(noisy_image_path),
            'noiseless_image': Path(noiseless_img_path),
            'noisy_signal': Path(noisy_signal_path),
            'noiseless_signal': Path(noiseless_signal_path),
            'noise': Path(noise_path)
        }
        
        # List all files for each path
        self.filenames = {
            key: sorted(list(path.glob('*'))) for key, path in self.paths.items()
        }
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return max(len(filenames) for filenames in self.filenames.values())
    
    def _load_and_preprocess_image(self, path: Path) -> np.ndarray:
        """Load and preprocess an image file.
        
        Args:
            path: Path to the image file.
            
        Returns:
            Preprocessed image as numpy array with shape (C, H, W).
        """
        img = Image.open(path).resize(self.image_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Handle grayscale images
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Convert to channels-first format (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        return img_array
    
    def _load_and_preprocess_signal(self, path: Path) -> np.ndarray:
        """Load and preprocess a signal file.
        
        Args:
            path: Path to the numpy signal file.
            
        Returns:
            Preprocessed signal as numpy array with shape (C, H, W).
        """
        npy_data = np.load(path)
        
        # Interpolate to target length if needed
        # Handle complex signals by interpolating real and imaginary parts separately
        if len(npy_data) != self.target_length:
            x_new = np.linspace(0, len(npy_data) - 1, self.target_length)
            x_old = np.arange(len(npy_data))
            if np.iscomplexobj(npy_data):
                real_interp = np.interp(x_new, x_old, npy_data.real)
                imag_interp = np.interp(x_new, x_old, npy_data.imag)
                # Use magnitude for visualization
                npy_data = np.abs(real_interp + 1j * imag_interp)
            else:
                npy_data = np.interp(x_new, x_old, npy_data.real)
        else:
            # Use magnitude for complex signals
            npy_data = np.abs(npy_data) if np.iscomplexobj(npy_data) else npy_data.real
        
        # Reshape to image format
        npy_data = npy_data.reshape(self.image_size[0], self.image_size[1])
        npy_data = np.stack([npy_data] * 3, axis=0)  # (C, H, W)
        
        return npy_data.astype(np.float32)
    
    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get a sample by index.
        
        Args:
            index: Sample index.
            
        Returns:
            Tuple of (inputs, targets) where each is a list of arrays.
        """
        # Load noisy and noiseless images
        noisy_img = self._load_and_preprocess_image(self.filenames['noisy_image'][index])
        noiseless_img = self._load_and_preprocess_image(self.filenames['noiseless_image'][index])
        
        # Load signals
        noisy_signal = self._load_and_preprocess_signal(self.filenames['noisy_signal'][index])
        noiseless_signal = self._load_and_preprocess_signal(self.filenames['noiseless_signal'][index])
        noise_data = self._load_and_preprocess_signal(self.filenames['noise'][index])
        
        # All modalities as inputs and their clean versions as targets
        all_inputs = [noisy_img, noisy_signal, noiseless_img, noiseless_signal, noise_data]
        all_targets = [noiseless_img, noiseless_signal, noiseless_img, noiseless_signal, noise_data]
        
        return all_inputs[:self.num_modalities], all_targets[:self.num_modalities]


class ImageFolderDataset:
    """Dataset for image classification from folder structure.
    
    Expects folder structure:
        root/
            class1/
                image1.png
                image2.png
                ...
            class2/
                image1.png
                ...
    
    Args:
        root: Root directory.
        image_size: Size to resize images to.
    """
    
    def __init__(
        self,
        root: str,
        image_size: Tuple[int, int] = (224, 224)
    ):
        self.root = Path(root)
        self.image_size = image_size
        
        # Find all classes (subdirectories)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for cls in self.classes:
            class_dir = self.root / cls
            for img_path in class_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[cls]))
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[cls]))
            for img_path in class_dir.glob('*.jpeg'):
                self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Get a sample by index.
        
        Args:
            index: Sample index.
            
        Returns:
            Tuple of (image, label).
        """
        img_path, label = self.samples[index]
        
        # Load and preprocess image
        img = Image.open(img_path).resize(self.image_size)
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Handle grayscale
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Convert to channels-first
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Normalize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_array = (img_array - mean) / std
        
        return img_array, label


def create_dataloader(
    dataset: DenoMAEDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42
) -> Iterator[Tuple[List[jnp.ndarray], List[jnp.ndarray]]]:
    """Create a data loader for DenoMAE training.
    
    Args:
        dataset: DenoMAEDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        
    Yields:
        Batches of (inputs, targets).
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Collect batch
        batch_inputs = []
        batch_targets = []
        
        for idx in batch_indices:
            inputs, targets = dataset[idx]
            batch_inputs.append(inputs)
            batch_targets.append(targets)
        
        # Stack into arrays
        num_modalities = len(batch_inputs[0])
        inputs_stacked = [
            jnp.stack([b[i] for b in batch_inputs], axis=0)
            for i in range(num_modalities)
        ]
        targets_stacked = [
            jnp.stack([b[i] for b in batch_targets], axis=0)
            for i in range(num_modalities)
        ]
        
        yield inputs_stacked, targets_stacked


def create_classification_loader(
    dataset: ImageFolderDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Create a data loader for classification.
    
    Args:
        dataset: ImageFolderDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        
    Yields:
        Batches of (images, labels).
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    if shuffle and seed is not None:
        np.random.seed(seed)
        np.random.shuffle(indices)
    elif shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Collect batch
        images = []
        labels = []
        
        for idx in batch_indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)
        
        yield jnp.stack(images, axis=0), jnp.array(labels)


def get_dataset_stats(dataset: DenoMAEDataset) -> Dict[str, Any]:
    """Compute dataset statistics.
    
    Args:
        dataset: DenoMAEDataset instance.
        
    Returns:
        Dictionary with dataset statistics.
    """
    n_samples = len(dataset)
    
    # Sample a few items to get shapes
    sample_inputs, sample_targets = dataset[0]
    
    stats = {
        'num_samples': n_samples,
        'num_modalities': len(sample_inputs),
        'input_shapes': [inp.shape for inp in sample_inputs],
        'target_shapes': [tgt.shape for tgt in sample_targets],
    }
    
    return stats
