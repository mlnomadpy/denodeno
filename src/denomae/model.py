"""DenoMAE model implementation using Flax NNX.

This module implements the DenoMAE (Denoising Masked Autoencoder) model
for multimodal signal processing.
"""

from typing import Tuple, Sequence
import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.encoder_decoder import TransformerEncoder, TransformerDecoder


class PatchEmbedding(nnx.Module):
    """Patch embedding layer for images.
    
    Converts an image into a sequence of patch embeddings using a convolutional layer.
    
    Args:
        img_size: Size of the input image (assumed square).
        patch_size: Size of each patch.
        in_chans: Number of input channels.
        embed_dim: Embedding dimension.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_chans: int = 3, 
        embed_dim: int = 768,
        rngs: nnx.Rngs = None
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Convolution for patch projection
        self.proj = nnx.Conv(
            in_features=in_chans,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding='VALID',
            rngs=rngs
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (batch, height, width, channels) or 
               (batch, channels, height, width).
               
        Returns:
            Patch embeddings of shape (batch, n_patches, embed_dim).
        """
        # Assume input is (B, C, H, W) - convert to (B, H, W, C) for JAX Conv
        if x.shape[1] == 3:  # Assume channels-first format
            x = jnp.transpose(x, (0, 2, 3, 1))  # (B, H, W, C)
        
        x = self.proj(x)  # (B, H', W', embed_dim)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.embed_dim)  # (B, n_patches, embed_dim)
        return x


class DenoMAE(nnx.Module):
    """Denoising Masked Autoencoder for multimodal signal processing.
    
    Args:
        num_modalities: Number of input modalities.
        img_size: Size of the input image (assumed square).
        patch_size: Size of the patches to be extracted from the image.
        in_chans: Number of input channels.
        mask_ratio: Ratio of patches to be masked.
        embed_dim: Dimension of the token embeddings.
        encoder_depth: Number of transformer encoder layers.
        decoder_depth: Number of transformer decoder layers.
        num_heads: Number of attention heads in transformer layers.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        num_modalities: int, 
        img_size: int = 224, 
        patch_size: int = 16,
        in_chans: int = 3, 
        mask_ratio: float = 0.75, 
        embed_dim: int = 768, 
        encoder_depth: int = 12,
        decoder_depth: int = 4, 
        num_heads: int = 12,
        rngs: nnx.Rngs = None
    ):
        self.num_modalities = num_modalities
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2
        
        # Patch embeddings for each modality - using nnx.List
        self.patch_embeds = nnx.List([
            PatchEmbedding(img_size, patch_size, in_chans, embed_dim, rngs=rngs)
            for _ in range(num_modalities)
        ])
        
        # CLS token and positional embeddings
        self.cls_token = nnx.Param(jnp.zeros((1, 1, embed_dim)))
        self.pos_embed = nnx.Param(jnp.zeros((1, self.n_patches + 1, embed_dim)))
        
        # Encoder
        self.encoder = TransformerEncoder(embed_dim, encoder_depth, num_heads, rngs=rngs)
        
        # Decoders for each modality - using nnx.List
        self.decoders = nnx.List([
            TransformerDecoder(embed_dim, decoder_depth, num_heads, rngs=rngs)
            for _ in range(num_modalities)
        ])
        
        # Output heads for each modality - using nnx.List
        self.modality_heads = nnx.List([
            nnx.Linear(embed_dim, patch_size**2 * in_chans, rngs=rngs)
            for _ in range(num_modalities)
        ])
    
    def random_masking(
        self, 
        x: jnp.ndarray, 
        key: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Perform random masking on the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, embed_dim).
            key: JAX random key for masking.
            
        Returns:
            Tuple of:
                - Masked input tensor
                - Binary mask (1 for masked, 0 for unmasked)
                - Indices for restoring the original order
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Generate random noise
        noise = jax.random.uniform(key, (N, L))  # noise in [0, 1]
        
        # Sort noise to get shuffle indices
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)
        
        # Keep the first len_keep indices
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather kept tokens
        # Use advanced indexing with vmap for efficiency
        def gather_tokens(x_single, ids):
            return x_single[ids]
        
        x_masked = jax.vmap(gather_tokens)(x, ids_keep)
        
        # Create mask: 0 for kept, 1 for removed
        mask = jnp.ones((N, L))
        mask = mask.at[:, :len_keep].set(0)
        
        # Unshuffle to get mask in original order
        def unshuffle_mask(mask_single, ids):
            return mask_single[jnp.argsort(ids)]
        
        mask = jax.vmap(unshuffle_mask)(
            jnp.ones((N, L)).at[:, :len_keep].set(0), 
            ids_shuffle
        )
        
        return x_masked, mask, ids_restore
    
    def __call__(
        self, 
        inputs: Sequence[jnp.ndarray], 
        key: jax.random.PRNGKey = None
    ) -> Tuple[Sequence[jnp.ndarray], Sequence[jnp.ndarray]]:
        """Forward pass of the DenoMAE model.
        
        Args:
            inputs: List of input tensors, one for each modality.
                   Each tensor should have shape (batch_size, in_chans, img_size, img_size).
            key: JAX random key for masking.
            
        Returns:
            Tuple of:
                - List of reconstructed outputs, one for each modality.
                  Each tensor has shape (batch_size, in_chans, img_size, img_size).
                - List of binary masks, one for each modality.
                  Each tensor has shape (batch_size, num_patches).
        """
        assert len(inputs) == self.num_modalities, \
            f"Expected {self.num_modalities} inputs, but got {len(inputs)}"
        
        if key is None:
            key = jax.random.PRNGKey(0)
        
        masked_embeddings = []
        masks = []
        ids_restores = []
        
        # Generate keys for each modality
        keys = jax.random.split(key, self.num_modalities)
        
        for i, x in enumerate(inputs):
            # Patch embedding
            x = self.patch_embeds[i](x)
            # Add positional embedding (excluding CLS position)
            x = x + self.pos_embed[..., 1:, :]
            # Random masking
            x_masked, mask, ids_restore = self.random_masking(x, keys[i])
            masked_embeddings.append(x_masked)
            masks.append(mask)
            ids_restores.append(ids_restore)
        
        # Concatenate masked embeddings from all modalities
        x_masked = jnp.concatenate(masked_embeddings, axis=1)
        
        # Add CLS token
        cls_token = self.cls_token[...] + self.pos_embed[..., :1, :]
        cls_tokens = jnp.broadcast_to(cls_token, (x_masked.shape[0], 1, self.embed_dim))
        x_masked = jnp.concatenate([cls_tokens, x_masked], axis=1)
        
        # Encode
        encoded = self.encoder(x_masked)
        
        # Decode each modality
        decoded_outputs = []
        segment_length = (encoded.shape[1] - 1) // self.num_modalities
        
        for i in range(self.num_modalities):
            start_idx = 1 + i * segment_length
            end_idx = 1 + (i + 1) * segment_length
            decoded = self.decoders[i](encoded[:, start_idx:end_idx])
            decoded_outputs.append(decoded)
        
        # Reconstruct each modality
        reconstructions = []
        for i, decoded in enumerate(decoded_outputs):
            # Project to pixel space
            rec = self.modality_heads[i](decoded)
            rec = rec.reshape(
                rec.shape[0], -1, self.patch_size, self.patch_size, self.in_chans
            )
            
            # Create full reconstruction with zeros for masked positions
            batch_size = rec.shape[0]
            n_keep = int(self.n_patches * (1 - self.mask_ratio))
            
            full_rec = jnp.zeros(
                (batch_size, self.n_patches, self.patch_size, self.patch_size, self.in_chans)
            )
            
            # Fill in the kept positions
            full_rec = full_rec.at[:, :n_keep].set(rec)
            
            # Unshuffle to original order
            def unshuffle_rec(rec_single, ids):
                return rec_single[jnp.argsort(ids)]
            
            full_rec = jax.vmap(unshuffle_rec)(full_rec, ids_restores[i])
            
            # Reshape to image format
            h_patches = w_patches = self.img_size // self.patch_size
            full_rec = full_rec.reshape(
                batch_size, h_patches, w_patches, self.patch_size, self.patch_size, self.in_chans
            )
            full_rec = full_rec.transpose(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
            full_rec = full_rec.reshape(
                batch_size, self.in_chans, self.img_size, self.img_size
            )
            
            reconstructions.append(full_rec)
        
        return reconstructions, masks


class FineTunedDenoMAE(nnx.Module):
    """Fine-tuned DenoMAE model for classification.
    
    Args:
        backbone: Pretrained DenoMAE model.
        num_classes: Number of output classes.
        freeze_encoder: Whether to freeze the encoder weights.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        backbone: DenoMAE, 
        num_classes: int, 
        freeze_encoder: bool = True,
        rngs: nnx.Rngs = None
    ):
        self.backbone = backbone
        self.freeze_encoder = freeze_encoder
        
        # Classification head
        embed_dim = backbone.embed_dim
        self.classifier = nnx.Sequential(
            nnx.Linear(embed_dim, 512, rngs=rngs),
            lambda x: nnx.relu(x),
            nnx.Dropout(rate=0.5, rngs=rngs),
            nnx.Linear(512, num_classes, rngs=rngs)
        )
    
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Forward pass for classification.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width).
            deterministic: Whether to use deterministic mode (no dropout).
            
        Returns:
            Logits of shape (batch, num_classes).
        """
        # Patch embedding
        x = self.backbone.patch_embeds[0](x)
        # Add positional embedding
        x = x + self.backbone.pos_embed[..., 1:, :]
        # Encode
        x = self.backbone.encoder(x)
        # Global average pooling
        x = jnp.mean(x, axis=1)
        # Classify
        x = self.classifier(x)
        return x
