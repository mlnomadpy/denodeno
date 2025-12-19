"""Tests for DenoMAE model."""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.model import PatchEmbedding, DenoMAE, FineTunedDenoMAE


class TestPatchEmbedding:
    """Tests for PatchEmbedding module."""
    
    def test_patch_embedding_output_shape(self):
        """Test patch embedding output shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        patch_embed = PatchEmbedding(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            rngs=rngs
        )
        
        # Input: (batch, channels, height, width)
        x = jax.random.normal(key, (2, 3, 224, 224))
        y = patch_embed(x)
        
        expected_patches = (224 // 16) ** 2  # 196
        assert y.shape == (2, expected_patches, 768), f"Expected (2, 196, 768), got {y.shape}"
    
    def test_patch_embedding_different_sizes(self):
        """Test patch embedding with different image sizes."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for img_size, patch_size in [(224, 16), (256, 32), (128, 8)]:
            patch_embed = PatchEmbedding(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=768,
                rngs=rngs
            )
            
            x = jax.random.normal(key, (2, 3, img_size, img_size))
            y = patch_embed(x)
            
            expected_patches = (img_size // patch_size) ** 2
            assert y.shape == (2, expected_patches, 768)


class TestDenoMAE:
    """Tests for DenoMAE model."""
    
    def test_denomae_output_shapes(self):
        """Test DenoMAE outputs have correct shapes."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        num_modalities = 3
        img_size = 224
        patch_size = 16
        
        model = DenoMAE(
            num_modalities=num_modalities,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            mask_ratio=0.75,
            embed_dim=768,
            encoder_depth=4,  # Reduced for testing
            decoder_depth=2,
            num_heads=12,
            rngs=rngs
        )
        
        # Create inputs for each modality
        batch_size = 2
        inputs = [
            jax.random.normal(key, (batch_size, 3, img_size, img_size))
            for _ in range(num_modalities)
        ]
        
        reconstructions, masks = model(inputs, key)
        
        # Check number of outputs
        assert len(reconstructions) == num_modalities
        assert len(masks) == num_modalities
        
        # Check reconstruction shapes
        for rec in reconstructions:
            assert rec.shape == (batch_size, 3, img_size, img_size)
        
        # Check mask shapes
        n_patches = (img_size // patch_size) ** 2
        for mask in masks:
            assert mask.shape == (batch_size, n_patches)
    
    def test_denomae_different_modalities(self):
        """Test DenoMAE with different number of modalities."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        for num_modalities in [1, 3, 5]:
            model = DenoMAE(
                num_modalities=num_modalities,
                img_size=224,
                patch_size=16,
                encoder_depth=2,
                decoder_depth=2,
                rngs=rngs
            )
            
            inputs = [
                jax.random.normal(key, (2, 3, 224, 224))
                for _ in range(num_modalities)
            ]
            
            reconstructions, masks = model(inputs, key)
            
            assert len(reconstructions) == num_modalities
            assert len(masks) == num_modalities
    
    def test_denomae_masking(self):
        """Test that masking produces correct mask ratio."""
        key = jax.random.PRNGKey(2)
        rngs = nnx.Rngs(key)
        
        mask_ratio = 0.75
        model = DenoMAE(
            num_modalities=1,
            img_size=224,
            patch_size=16,
            mask_ratio=mask_ratio,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        inputs = [jax.random.normal(key, (4, 3, 224, 224))]
        _, masks = model(inputs, key)
        
        # Check approximate mask ratio
        actual_ratio = jnp.mean(masks[0])
        # Allow some tolerance due to integer rounding
        assert abs(actual_ratio - mask_ratio) < 0.1
    
    def test_denomae_uses_nnx_list(self):
        """Test that DenoMAE uses nnx.List for collections."""
        key = jax.random.PRNGKey(3)
        rngs = nnx.Rngs(key)
        
        model = DenoMAE(
            num_modalities=3,
            img_size=224,
            patch_size=16,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        assert isinstance(model.patch_embeds, nnx.List)
        assert isinstance(model.decoders, nnx.List)
        assert isinstance(model.modality_heads, nnx.List)


class TestFineTunedDenoMAE:
    """Tests for FineTunedDenoMAE model."""
    
    def test_finetuned_output_shape(self):
        """Test fine-tuned model output shape."""
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        
        # Create backbone
        backbone = DenoMAE(
            num_modalities=1,
            img_size=224,
            patch_size=16,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        num_classes = 10
        model = FineTunedDenoMAE(
            backbone=backbone,
            num_classes=num_classes,
            freeze_encoder=True,
            rngs=rngs
        )
        
        x = jax.random.normal(key, (4, 3, 224, 224))
        logits = model(x)
        
        assert logits.shape == (4, num_classes)
    
    def test_finetuned_different_classes(self):
        """Test fine-tuned model with different number of classes."""
        key = jax.random.PRNGKey(1)
        rngs = nnx.Rngs(key)
        
        backbone = DenoMAE(
            num_modalities=1,
            img_size=224,
            patch_size=16,
            encoder_depth=2,
            decoder_depth=2,
            rngs=rngs
        )
        
        for num_classes in [5, 10, 15]:
            model = FineTunedDenoMAE(
                backbone=backbone,
                num_classes=num_classes,
                rngs=rngs
            )
            
            x = jax.random.normal(key, (2, 3, 224, 224))
            logits = model(x)
            
            assert logits.shape == (2, num_classes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
