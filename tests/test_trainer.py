"""Tests for training utilities."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.model import DenoMAE
from src.training.trainer import (
    create_train_state,
    mse_loss,
    train_step,
)


class TestTrainState:
    """Tests for training state creation."""
    
    def test_create_train_state(self):
        """Test train state creation."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        model = DenoMAE(
            num_modalities=1,
            img_size=224,
            patch_size=16,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        state = create_train_state(model, learning_rate=1e-4)
        
        assert state is not None
        assert state.model is model


class TestMSELoss:
    """Tests for MSE loss computation."""
    
    def test_mse_loss_zero_for_equal(self):
        """Test MSE loss is zero when predictions equal targets."""
        predictions = [jnp.ones((2, 3, 224, 224))]
        targets = [jnp.ones((2, 3, 224, 224))]
        masks = [jnp.zeros((2, 196))]  # No masking
        
        loss = mse_loss(predictions, targets, masks, patch_size=16, image_size=224)
        
        assert jnp.allclose(loss, 0.0)
    
    def test_mse_loss_positive_for_different(self):
        """Test MSE loss is positive when predictions differ from targets."""
        predictions = [jnp.zeros((2, 3, 224, 224))]
        targets = [jnp.ones((2, 3, 224, 224))]
        masks = [jnp.zeros((2, 196))]
        
        loss = mse_loss(predictions, targets, masks, patch_size=16, image_size=224)
        
        assert loss > 0
    
    def test_mse_loss_multiple_modalities(self):
        """Test MSE loss with multiple modalities."""
        num_modalities = 3
        predictions = [jnp.ones((2, 3, 224, 224)) * i for i in range(num_modalities)]
        targets = [jnp.zeros((2, 3, 224, 224)) for _ in range(num_modalities)]
        masks = [jnp.zeros((2, 196)) for _ in range(num_modalities)]
        
        loss = mse_loss(predictions, targets, masks, patch_size=16, image_size=224)
        
        assert loss > 0


class TestTrainStep:
    """Tests for training step."""
    
    def test_train_step_updates_loss(self):
        """Test that train step produces a loss."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        model = DenoMAE(
            num_modalities=1,
            img_size=224,
            patch_size=16,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        state = create_train_state(model, learning_rate=1e-4)
        
        # Create dummy data
        inputs = [jax.random.normal(key, (2, 3, 224, 224))]
        targets = [jax.random.normal(key, (2, 3, 224, 224))]
        
        _, metrics = train_step(
            state, inputs, targets, key, 
            patch_size=16, image_size=224
        )
        
        assert 'loss' in metrics
        assert metrics['loss'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
