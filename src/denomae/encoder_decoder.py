"""Transformer Encoder and Decoder components using Flax NNX.

This module implements the transformer encoder and decoder blocks using the 
latest Flax NNX API with nnx.List for module collections.
"""

from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    """Multi-Layer Perceptron block.
    
    Args:
        in_features: Input feature dimension.
        hidden_features: Hidden layer dimension.
        rngs: Random number generator state.
    """
    
    def __init__(self, in_features: int, hidden_features: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, in_features, rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of MLP.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features).
            
        Returns:
            Output tensor of shape (batch, seq_len, features).
        """
        x = self.fc1(x)
        x = nnx.gelu(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nnx.Module):
    """Transformer encoder block with self-attention and MLP.
    
    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of hidden dimension to embedding dimension in MLP.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0,
        rngs: nnx.Rngs = None
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm1 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            decode=False,
            rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), rngs=rngs)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of transformer encoder block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        # Self-attention with residual connection
        normed = self.norm1(x)
        attn_out = self.attn(normed)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nnx.Module):
    """Transformer encoder with multiple encoder blocks.
    
    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of hidden dimension to embedding dimension in MLP.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        depth: int = 12, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0,
        rngs: nnx.Rngs = None
    ):
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Use nnx.List for module collections (best practice from latest Flax NNX)
        self.blocks = nnx.List([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, rngs=rngs)
            for _ in range(depth)
        ])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all encoder blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x


class TransformerDecoder(nnx.Module):
    """Transformer decoder with multiple encoder blocks.
    
    Note: This uses the same architecture as encoder blocks (self-attention only),
    following the original MAE design where the decoder uses self-attention
    rather than cross-attention.
    
    Args:
        embed_dim: Embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of hidden dimension to embedding dimension in MLP.
        rngs: Random number generator state.
    """
    
    def __init__(
        self, 
        embed_dim: int = 768, 
        depth: int = 4, 
        num_heads: int = 12, 
        mlp_ratio: float = 4.0,
        rngs: nnx.Rngs = None
    ):
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Use nnx.List for module collections (best practice from latest Flax NNX)
        self.blocks = nnx.List([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, rngs=rngs)
            for _ in range(depth)
        ])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through all decoder blocks.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim).
        """
        for block in self.blocks:
            x = block(x)
        return x
