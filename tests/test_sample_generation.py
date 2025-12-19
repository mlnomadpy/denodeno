"""Tests for sample generation module."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import os

from src.data.sample_generation import (
    awgn,
    gmsk_modulate,
    generate_image,
    MOD_TYPES,
)


class TestAWGN:
    """Tests for AWGN function."""
    
    def test_awgn_output_shape(self):
        """Test AWGN preserves signal shape."""
        key = jax.random.PRNGKey(0)
        signal = jax.random.normal(key, (1024,)) + 1j * jax.random.normal(key, (1024,))
        
        noisy_signal, noise = awgn(signal, snr_dB=10.0, key=key)
        
        assert noisy_signal.shape == signal.shape
        assert noise.shape == signal.shape
    
    def test_awgn_snr_effect(self):
        """Test that higher SNR produces less noise."""
        key = jax.random.PRNGKey(1)
        signal = jax.random.normal(key, (1024,)) + 1j * jax.random.normal(key, (1024,))
        
        _, noise_low_snr = awgn(signal, snr_dB=0.0, key=key)
        _, noise_high_snr = awgn(signal, snr_dB=20.0, key=key)
        
        noise_power_low = jnp.mean(jnp.abs(noise_low_snr) ** 2)
        noise_power_high = jnp.mean(jnp.abs(noise_high_snr) ** 2)
        
        assert noise_power_high < noise_power_low
    
    def test_awgn_complex_output(self):
        """Test AWGN produces complex output."""
        key = jax.random.PRNGKey(2)
        signal = jnp.ones(100) + 1j * jnp.ones(100)
        
        noisy_signal, noise = awgn(signal, snr_dB=10.0, key=key)
        
        assert jnp.iscomplexobj(noisy_signal)
        assert jnp.iscomplexobj(noise)


class TestGMSKModulate:
    """Tests for GMSK modulation."""
    
    def test_gmsk_output_length(self):
        """Test GMSK output length."""
        bits = jnp.array([0, 1, 0, 1, 1, 0, 0, 1])
        samples_per_symbol = 8
        
        signal = gmsk_modulate(bits, bt_product=0.3, samples_per_symbol=samples_per_symbol)
        
        # Output length should be bits * samples_per_symbol
        expected_length = len(bits) * samples_per_symbol
        assert len(signal) == expected_length
    
    def test_gmsk_unit_magnitude(self):
        """Test GMSK signal has approximately unit magnitude."""
        bits = jax.random.randint(jax.random.PRNGKey(0), (100,), 0, 2)
        
        signal = gmsk_modulate(bits, bt_product=0.3, samples_per_symbol=8)
        
        magnitudes = jnp.abs(signal)
        # GMSK should have unit envelope
        assert jnp.allclose(magnitudes, 1.0, atol=1e-5)
    
    def test_gmsk_complex_output(self):
        """Test GMSK produces complex output."""
        bits = jnp.array([0, 1, 0, 1])
        
        signal = gmsk_modulate(bits)
        
        assert jnp.iscomplexobj(signal)


class TestModTypes:
    """Tests for modulation type configurations."""
    
    def test_mod_types_exist(self):
        """Test that all expected modulation types exist."""
        expected_types = [
            'OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK',
            '4PAM', 'DQPSK', '16PAM', 'GMSK', 'BPSK', 'QPSK',
            '8PSK', '16QAM', '4FSK'
        ]
        
        for mod_type in expected_types:
            assert mod_type in MOD_TYPES, f"Missing modulation type: {mod_type}"
    
    def test_mod_types_have_constellation(self):
        """Test that all mod types have valid constellation."""
        for mod_type, (constellation, order) in MOD_TYPES.items():
            assert len(constellation) > 0, f"Empty constellation for {mod_type}"
            assert order > 0, f"Invalid order for {mod_type}"
    
    def test_constellation_sizes(self):
        """Test constellation sizes match expected."""
        expected_sizes = {
            'OOK': 2,
            '4ASK': 4,
            '8ASK': 8,
            'QPSK': 4,
            '8PSK': 8,
            '16QAM': 16,
        }
        
        for mod_type, expected_size in expected_sizes.items():
            constellation, _ = MOD_TYPES[mod_type]
            assert len(constellation) == expected_size, \
                f"Expected {expected_size} symbols for {mod_type}, got {len(constellation)}"


class TestGenerateImage:
    """Tests for image generation."""
    
    def test_generate_image_creates_file(self):
        """Test that generate_image creates an image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple signal
            signal = np.array([0.5 + 0.5j, -0.5 + 0.5j, -0.5 - 0.5j, 0.5 - 0.5j])
            signal = np.repeat(signal, 256)  # Repeat to have enough points
            
            generate_image(
                signal=signal,
                image_size=(224, 224),
                image_dir=tmpdir,
                image_name='test_image'
            )
            
            output_path = os.path.join(tmpdir, 'test_image.png')
            assert os.path.exists(output_path), "Image file not created"
    
    def test_generate_image_correct_size(self):
        """Test that generated image has correct size."""
        from PIL import Image
        
        with tempfile.TemporaryDirectory() as tmpdir:
            signal = np.random.randn(1024) + 1j * np.random.randn(1024)
            image_size = (224, 224)
            
            generate_image(
                signal=signal,
                image_size=image_size,
                image_dir=tmpdir,
                image_name='test_size'
            )
            
            output_path = os.path.join(tmpdir, 'test_size.png')
            img = Image.open(output_path)
            
            assert img.size == image_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
