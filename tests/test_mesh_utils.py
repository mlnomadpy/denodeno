"""Tests for mesh utilities."""

import pytest
import jax
import jax.numpy as jnp

from src.denomae.mesh_utils import (
    create_device_mesh,
    get_data_parallel_sharding,
    get_replicated_sharding,
    DataParallelTrainer,
)


class TestDeviceMesh:
    """Tests for device mesh creation."""
    
    def test_create_mesh(self):
        """Test mesh creation with default settings."""
        mesh = create_device_mesh()
        
        assert mesh is not None
        assert 'data' in mesh.axis_names
    
    def test_mesh_shape(self):
        """Test mesh shape matches device count."""
        mesh = create_device_mesh()
        
        num_devices = len(jax.devices())
        assert mesh.size == num_devices


class TestSharding:
    """Tests for sharding specifications."""
    
    def test_data_parallel_sharding(self):
        """Test data parallel sharding creation."""
        mesh = create_device_mesh()
        sharding = get_data_parallel_sharding(mesh)
        
        assert sharding is not None
    
    def test_replicated_sharding(self):
        """Test replicated sharding creation."""
        mesh = create_device_mesh()
        sharding = get_replicated_sharding(mesh)
        
        assert sharding is not None


class TestDataParallelTrainer:
    """Tests for DataParallelTrainer."""
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        trainer = DataParallelTrainer()
        
        assert trainer is not None
        assert trainer.mesh is not None
    
    def test_trainer_with_mesh(self):
        """Test trainer creation with custom mesh."""
        mesh = create_device_mesh()
        trainer = DataParallelTrainer(mesh=mesh)
        
        assert trainer.mesh is mesh
    
    def test_shard_batch(self):
        """Test batch sharding."""
        trainer = DataParallelTrainer()
        
        batch = jnp.ones((8, 3, 224, 224))
        sharded = trainer.shard_batch(batch)
        
        assert sharded is not None
        assert sharded.shape == batch.shape
    
    def test_shard_batch_list(self):
        """Test sharding list of arrays."""
        trainer = DataParallelTrainer()
        
        batch = [jnp.ones((8, 3, 224, 224)) for _ in range(3)]
        sharded = trainer.shard_batch(batch)
        
        assert len(sharded) == len(batch)
        for s, b in zip(sharded, batch):
            assert s.shape == b.shape
    
    def test_shard_batch_dict(self):
        """Test sharding dictionary of arrays."""
        trainer = DataParallelTrainer()
        
        batch = {
            'images': jnp.ones((8, 3, 224, 224)),
            'labels': jnp.ones((8,))
        }
        sharded = trainer.shard_batch(batch)
        
        assert set(sharded.keys()) == set(batch.keys())
        for k in batch:
            assert sharded[k].shape == batch[k].shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
