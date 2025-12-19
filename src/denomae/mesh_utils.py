"""Mesh utilities for distributed training on TPU v5-8.

This module provides utilities for setting up JAX device mesh for
pure data parallel training on TPU v5-8 (8 chips).
"""

from typing import Tuple, Any, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils


def create_device_mesh(
    mesh_shape: Optional[Tuple[int, ...]] = None,
    axis_names: Tuple[str, ...] = ('data',)
) -> Mesh:
    """Create a device mesh for distributed training.
    
    For TPU v5-8, the default configuration is 8 devices in data parallel mode.
    
    Args:
        mesh_shape: Shape of the device mesh. Default is (num_devices,) for pure data parallel.
        axis_names: Names for mesh axes. Default is ('data',) for pure data parallel.
        
    Returns:
        JAX Mesh object for distributed computation.
    """
    devices = jax.devices()
    num_devices = len(devices)
    
    if mesh_shape is None:
        mesh_shape = (num_devices,)
    
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    mesh = Mesh(device_mesh, axis_names)
    
    return mesh


def get_data_parallel_sharding(mesh: Mesh, batch_axis: int = 0) -> NamedSharding:
    """Get sharding spec for data parallel distribution.
    
    Shards data along the batch dimension across all devices.
    
    Args:
        mesh: Device mesh.
        batch_axis: Axis along which to shard (default 0 for batch dimension).
        
    Returns:
        NamedSharding for data parallel distribution.
    """
    # Create partition spec with 'data' axis for the batch dimension
    # Other dimensions are replicated (None)
    pspec = PartitionSpec('data')
    return NamedSharding(mesh, pspec)


def get_replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Get sharding spec for replicated data (e.g., model parameters).
    
    Args:
        mesh: Device mesh.
        
    Returns:
        NamedSharding for replicated data.
    """
    pspec = PartitionSpec()
    return NamedSharding(mesh, pspec)


def shard_batch(batch: Any, mesh: Mesh) -> Any:
    """Shard a batch of data across devices.
    
    Args:
        batch: Input batch (can be array, tuple, or dict).
        mesh: Device mesh.
        
    Returns:
        Sharded batch.
    """
    sharding = get_data_parallel_sharding(mesh)
    
    if isinstance(batch, dict):
        return {k: jax.device_put(v, sharding) for k, v in batch.items()}
    elif isinstance(batch, (tuple, list)):
        return type(batch)(jax.device_put(x, sharding) for x in batch)
    else:
        return jax.device_put(batch, sharding)


def replicate_params(params: Any, mesh: Mesh) -> Any:
    """Replicate parameters across all devices.
    
    Args:
        params: Model parameters.
        mesh: Device mesh.
        
    Returns:
        Replicated parameters.
    """
    sharding = get_replicated_sharding(mesh)
    
    def replicate_leaf(x):
        if isinstance(x, jnp.ndarray):
            return jax.device_put(x, sharding)
        return x
    
    return jax.tree_util.tree_map(replicate_leaf, params)


def print_mesh_info(mesh: Mesh) -> None:
    """Print information about the device mesh.
    
    Args:
        mesh: Device mesh.
    """
    print(f"Device mesh shape: {mesh.shape}")
    print(f"Device mesh axes: {mesh.axis_names}")
    print(f"Number of devices: {mesh.size}")
    print(f"Devices: {mesh.devices}")


class DataParallelTrainer:
    """Helper class for data parallel training on TPU.
    
    Provides utilities for:
    - Automatic sharding of batches
    - Gradient aggregation across devices
    - Synchronized training step
    
    Args:
        mesh: Device mesh for distribution.
    """
    
    def __init__(self, mesh: Optional[Mesh] = None):
        if mesh is None:
            mesh = create_device_mesh()
        self.mesh = mesh
        self.data_sharding = get_data_parallel_sharding(mesh)
        self.replicated_sharding = get_replicated_sharding(mesh)
    
    def shard_batch(self, batch: Any) -> Any:
        """Shard a batch across devices."""
        return shard_batch(batch, self.mesh)
    
    def replicate_params(self, params: Any) -> Any:
        """Replicate parameters across devices."""
        return replicate_params(params, self.mesh)
    
    def pmap_train_step(self, train_step_fn):
        """Wrap a training step function for data parallel execution.
        
        Args:
            train_step_fn: Function that takes (state, batch) and returns (state, metrics).
            
        Returns:
            Wrapped function that handles sharding automatically.
        """
        @jax.jit
        def wrapped_train_step(state, batch):
            with self.mesh:
                return train_step_fn(state, batch)
        
        return wrapped_train_step


def setup_tpu_v5_8():
    """Setup function specifically for TPU v5-8 (8 chips).
    
    Returns:
        Tuple of (mesh, trainer) configured for TPU v5-8.
    """
    # TPU v5-8 has 8 chips
    mesh = create_device_mesh(mesh_shape=(8,), axis_names=('data',))
    trainer = DataParallelTrainer(mesh)
    
    print("TPU v5-8 setup complete:")
    print_mesh_info(mesh)
    
    return mesh, trainer
