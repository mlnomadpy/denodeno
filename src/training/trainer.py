"""Training utilities for DenoMAE using Flax NNX and JAX.

This module provides training utilities including train state management,
training steps, and evaluation functions.
"""

from typing import Tuple, Any, Dict, List, Callable, Optional
from functools import partial
import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from src.denomae.model import DenoMAE
from src.denomae.mesh_utils import DataParallelTrainer


@dataclasses.dataclass
class TrainState:
    """Train state for DenoMAE training.
    
    Attributes:
        model: The DenoMAE model.
        optimizer: The nnx.Optimizer instance.
    """
    model: DenoMAE
    optimizer: nnx.Optimizer


def create_train_state(
    model: DenoMAE,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05
) -> TrainState:
    """Create training state.
    
    Args:
        model: DenoMAE model.
        learning_rate: Learning rate.
        weight_decay: Weight decay for AdamW.
        
    Returns:
        Training state with optimizer.
    """
    # Create optimizer
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    # Create nnx.Optimizer with wrt argument specifying what to optimize
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    
    return TrainState(model=model, optimizer=optimizer)


def mse_loss(predictions: List[jnp.ndarray], targets: List[jnp.ndarray], masks: List[jnp.ndarray], patch_size: int, image_size: int) -> jnp.ndarray:
    """Compute MSE loss on masked regions.
    
    Args:
        predictions: List of predicted outputs.
        targets: List of target outputs.
        masks: List of binary masks.
        patch_size: Patch size.
        image_size: Image size.
        
    Returns:
        Total MSE loss.
    """
    total_loss = 0.0
    
    for pred, target, mask in zip(predictions, targets, masks):
        # Reshape mask to match image dimensions
        # mask shape: (batch, n_patches)
        n_patches_per_side = image_size // patch_size
        mask_reshaped = mask.reshape(mask.shape[0], 1, n_patches_per_side, n_patches_per_side)
        # Repeat for patch pixels and channels
        mask_reshaped = jnp.repeat(mask_reshaped, target.shape[1], axis=1)  # channels
        mask_reshaped = jnp.repeat(mask_reshaped, patch_size, axis=2)  # height
        mask_reshaped = jnp.repeat(mask_reshaped, patch_size, axis=3)  # width
        
        # Compute loss only on masked regions
        loss = jnp.mean((pred - target) ** 2 * (1 - mask_reshaped))
        total_loss += loss
    
    return total_loss


def train_step(
    state: TrainState,
    inputs: List[jnp.ndarray],
    targets: List[jnp.ndarray],
    key: jax.random.PRNGKey,
    patch_size: int,
    image_size: int
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """Execute a single training step.
    
    Args:
        state: Training state.
        inputs: List of input tensors.
        targets: List of target tensors.
        key: Random key for masking.
        patch_size: Patch size.
        image_size: Image size.
        
    Returns:
        Tuple of (updated state, metrics dict).
    """
    model = state.model
    
    # Create a jitted loss and grad function
    @nnx.jit
    def compute_loss_and_update(model, optimizer, inputs, targets, key):
        def loss_fn(model):
            reconstructions, masks = model(inputs, key)
            loss = mse_loss(reconstructions, targets, masks, patch_size, image_size)
            return loss
        
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss
    
    loss = compute_loss_and_update(model, state.optimizer, inputs, targets, key)
    
    metrics = {'loss': loss}
    return state, metrics


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(
    model: DenoMAE,
    inputs: List[jnp.ndarray],
    targets: List[jnp.ndarray],
    patch_size: int,
    image_size: int,
    key: jax.random.PRNGKey = None
) -> Tuple[List[jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Execute a single evaluation step.
    
    Args:
        model: DenoMAE model.
        inputs: List of input tensors.
        targets: List of target tensors.
        patch_size: Patch size.
        image_size: Image size.
        key: Random key.
        
    Returns:
        Tuple of (reconstructions, metrics dict).
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    reconstructions, masks = model(inputs, key)
    loss = mse_loss(reconstructions, targets, masks, patch_size, image_size)
    
    metrics = {'loss': loss}
    return reconstructions, metrics


def train_epoch(
    state: TrainState,
    dataloader,
    patch_size: int,
    image_size: int,
    key: jax.random.PRNGKey,
    trainer: Optional[DataParallelTrainer] = None,
    log_interval: int = 50
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch.
    
    Args:
        state: Training state.
        dataloader: Data loader iterator.
        patch_size: Patch size.
        image_size: Image size.
        key: Random key.
        trainer: Optional data parallel trainer for TPU.
        log_interval: Interval for logging.
        
    Returns:
        Tuple of (updated state, epoch metrics).
    """
    epoch_metrics = []
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        key, subkey = jax.random.split(key)
        
        # Shard batch if using data parallel
        if trainer is not None:
            inputs = [trainer.shard_batch(x) for x in inputs]
            targets = [trainer.shard_batch(x) for x in targets]
        
        state, metrics = train_step(
            state, inputs, targets, subkey, patch_size, image_size
        )
        epoch_metrics.append(metrics)
        
        if batch_idx % log_interval == 0:
            print(f"Step {batch_idx}, Loss: {metrics['loss']:.4f}")
    
    # Aggregate metrics
    avg_metrics = {
        k: float(jnp.mean(jnp.array([m[k] for m in epoch_metrics])))
        for k in epoch_metrics[0].keys()
    }
    
    return state, avg_metrics


def evaluate(
    model: DenoMAE,
    dataloader,
    patch_size: int,
    image_size: int,
    key: jax.random.PRNGKey,
    trainer: Optional[DataParallelTrainer] = None
) -> Tuple[List[jnp.ndarray], Dict[str, float]]:
    """Evaluate model on dataset.
    
    Args:
        model: DenoMAE model.
        dataloader: Data loader iterator.
        patch_size: Patch size.
        image_size: Image size.
        key: Random key.
        trainer: Optional data parallel trainer.
        
    Returns:
        Tuple of (last reconstructions, metrics).
    """
    all_metrics = []
    last_reconstructions = None
    
    for inputs, targets in dataloader:
        key, subkey = jax.random.split(key)
        
        if trainer is not None:
            inputs = [trainer.shard_batch(x) for x in inputs]
            targets = [trainer.shard_batch(x) for x in targets]
        
        reconstructions, metrics = eval_step(
            model, inputs, targets, patch_size, image_size, subkey
        )
        all_metrics.append(metrics)
        last_reconstructions = reconstructions
    
    # Aggregate metrics
    avg_metrics = {
        k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
        for k in all_metrics[0].keys()
    }
    
    return last_reconstructions, avg_metrics


def save_checkpoint(
    state: TrainState,
    path: str,
    epoch: int,
    best_loss: float
) -> None:
    """Save training checkpoint.
    
    Args:
        state: Training state.
        path: Path to save checkpoint.
        epoch: Current epoch.
        best_loss: Best loss so far.
    """
    import pickle
    
    # Extract model state
    model_state = nnx.state(state.model)
    
    checkpoint = {
        'model_state': model_state,
        'epoch': epoch,
        'best_loss': best_loss
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: DenoMAE,
    path: str
) -> Tuple[DenoMAE, Dict[str, Any]]:
    """Load training checkpoint.
    
    Args:
        model: DenoMAE model to load state into.
        path: Path to checkpoint.
        
    Returns:
        Tuple of (model with loaded state, checkpoint metadata).
    """
    import pickle
    
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Update model state
    nnx.update(model, checkpoint['model_state'])
    
    metadata = {
        'epoch': checkpoint['epoch'],
        'best_loss': checkpoint['best_loss']
    }
    
    return model, metadata
