"""Tests for encoder/decoder components."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.encoder_decoder import (
    MLP,
    TransformerEncoderBlock,
    TransformerEncoder,
    TransformerDecoder,
)


class TestMLP:
    """Tests for MLP module."""
    
    def test_mlp_output_shape(self):
        """Test MLP preserves input shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        mlp = MLP(in_features=768, hidden_features=3072, rngs=rngs)
        
        x = jax.random.normal(key, (2, 196, 768))
        y = mlp(x)
        
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"
    
    def test_mlp_different_dims(self):
        """Test MLP with different dimensions."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for in_features in [256, 512, 1024]:
            mlp = MLP(in_features=in_features, hidden_features=in_features * 4, rngs=rngs)
            x = jax.random.normal(key, (4, 64, in_features))
            y = mlp(x)
            assert y.shape == x.shape


class TestTransformerEncoderBlock:
    """Tests for TransformerEncoderBlock."""
    
    def test_encoder_block_output_shape(self):
        """Test encoder block output shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        block = TransformerEncoderBlock(
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            rngs=rngs
        )
        
        x = jax.random.normal(key, (2, 197, 768))
        y = block(x)
        
        assert y.shape == x.shape
    
    def test_encoder_block_attention_heads(self):
        """Test encoder block with different attention heads."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for num_heads in [4, 8, 12]:
            block = TransformerEncoderBlock(
                embed_dim=768,
                num_heads=num_heads,
                rngs=rngs
            )
            x = jax.random.normal(key, (2, 100, 768))
            y = block(x)
            assert y.shape == x.shape


class TestTransformerEncoder:
    """Tests for TransformerEncoder."""
    
    def test_encoder_output_shape(self):
        """Test transformer encoder output shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        encoder = TransformerEncoder(
            embed_dim=768,
            depth=12,
            num_heads=12,
            rngs=rngs
        )
        
        x = jax.random.normal(key, (2, 197, 768))
        y = encoder(x)
        
        assert y.shape == x.shape
    
    def test_encoder_different_depths(self):
        """Test encoder with different depths."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for depth in [4, 8, 12]:
            encoder = TransformerEncoder(
                embed_dim=768,
                depth=depth,
                num_heads=12,
                rngs=rngs
            )
            x = jax.random.normal(key, (2, 64, 768))
            y = encoder(x)
            assert y.shape == x.shape
    
    def test_encoder_uses_nnx_list(self):
        """Test that encoder uses nnx.List for blocks."""
        key = jax.random.PRNGKey(2)
        rngs = nnx.Rngs(key)
        
        encoder = TransformerEncoder(embed_dim=768, depth=4, num_heads=12, rngs=rngs)
        
        assert isinstance(encoder.blocks, nnx.List)
        assert len(encoder.blocks) == 4


class TestTransformerDecoder:
    """Tests for TransformerDecoder."""
    
    def test_decoder_output_shape(self):
        """Test transformer decoder output shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        decoder = TransformerDecoder(
            embed_dim=768,
            depth=4,
            num_heads=12,
            rngs=rngs
        )
        
        x = jax.random.normal(key, (2, 49, 768))
        y = decoder(x)
        
        assert y.shape == x.shape
    
    def test_decoder_different_depths(self):
        """Test decoder with different depths."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for depth in [2, 4, 6]:
            decoder = TransformerDecoder(
                embed_dim=768,
                depth=depth,
                num_heads=12,
                rngs=rngs
            )
            x = jax.random.normal(key, (2, 49, 768))
            y = decoder(x)
            assert y.shape == x.shape
    
    def test_decoder_uses_nnx_list(self):
        """Test that decoder uses nnx.List for blocks."""
        key = jax.random.PRNGKey(2)
        rngs = nnx.Rngs(key)
        
        decoder = TransformerDecoder(embed_dim=768, depth=4, num_heads=12, rngs=rngs)
        
        assert isinstance(decoder.blocks, nnx.List)
        assert len(decoder.blocks) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
