"""Sample generation module using JAX for signal processing.

This module implements signal generation functions including AWGN (Additive White
Gaussian Noise) and GMSK (Gaussian Minimum Shift Keying) modulation using JAX
for efficient computation and TPU compatibility.
"""

import os
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


# Define modulation types and their corresponding constellation diagrams
MOD_TYPES: Dict[str, Tuple[np.ndarray, int]] = {
    'OOK': (np.array([0, 1]), 1),
    '4ASK': (np.array([-3, -1, 1, 3]), 2),
    '8ASK': (np.array([-7, -5, -3, -1, 1, 3, 5, 7]), 3),
    'OQPSK': (np.exp((np.arange(4) / 4) * 2 * np.pi * 1j + np.pi / 4), 2),
    'CPFSK': (np.exp(1j * 2 * np.pi * np.array([0.25, 0.75])), 1),
    'GFSK': (np.exp(1j * 2 * np.pi * np.array([0.25, 0.75])), 1),
    '4PAM': (np.array([-3, -1, 1, 3]), 2),
    'DQPSK': (np.exp(1j * np.array([0, np.pi/2, np.pi, -np.pi/2])), 2),
    '16PAM': (np.arange(-15, 16, 2), 4),
    'GMSK': (np.array([0, 1]), 1),
    'BPSK': (np.exp(1j * np.array([0, np.pi])), 1),
    'QPSK': (np.exp(1j * np.array([np.pi/4, 3*np.pi/4, -3*np.pi/4, -np.pi/4])), 2),
    '8PSK': (np.exp(1j * np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, 
                                   np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])), 3),
    '16QAM': (np.array([-3-3j, -3-1j, -3+1j, -3+3j, -1-3j, -1-1j, -1+1j, -1+3j,
                        1-3j, 1-1j, 1+1j, 1+3j, 3-3j, 3-1j, 3+1j, 3+3j]), 4),
    '4FSK': (np.exp(1j * 2 * np.pi * np.array([0.1, 0.2, 0.3, 0.4])), 2),
}


def awgn(
    signal: jnp.ndarray,
    snr_dB: float,
    key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Add Additive White Gaussian Noise to a signal.
    
    Args:
        signal: Input complex signal.
        snr_dB: Signal-to-noise ratio in dB.
        key: JAX random key for reproducibility.
        
    Returns:
        Tuple of (noisy_signal, noise).
    """
    snr = 10 ** (snr_dB / 10.0)
    power_signal = jnp.mean(jnp.abs(signal) ** 2)
    power_noise = power_signal / snr
    
    key1, key2 = jax.random.split(key)
    noise_real = jax.random.normal(key1, signal.shape)
    noise_imag = jax.random.normal(key2, signal.shape)
    noise = jnp.sqrt(power_noise / 2) * (noise_real + 1j * noise_imag)
    
    return signal + noise, noise


def gmsk_modulate(
    bits: jnp.ndarray,
    bt_product: float = 0.3,
    samples_per_symbol: int = 8
) -> jnp.ndarray:
    """Perform GMSK (Gaussian Minimum Shift Keying) modulation.
    
    This implementation uses a windowed sinc filter for pulse shaping, which
    approximates Gaussian pulse shaping used in standard GMSK. The filter is
    normalized to preserve signal energy.
    
    Note: The bt_product parameter is retained for API compatibility but the
    current implementation uses a fixed filter design that provides good
    spectral properties similar to GMSK with BT=0.3.
    
    Args:
        bits: Binary input bits (0 or 1).
        bt_product: Bandwidth-time product (retained for compatibility).
        samples_per_symbol: Number of samples per symbol.
        
    Returns:
        Complex GMSK modulated signal with length = len(bits) * samples_per_symbol.
    """
    h = 0.5  # Modulation index for MSK
    
    # Create pulse shaping filter using windowed sinc
    # This provides smooth frequency transitions similar to Gaussian pulse shaping
    filter_span = 8  # Filter spans 8 symbols
    filter_len = filter_span * samples_per_symbol + 1
    t = jnp.linspace(-filter_span/2, filter_span/2, filter_len)
    g = jnp.sinc(t) * jnp.hamming(filter_len)  # Windowed sinc for smooth transitions
    g = g / jnp.sum(g)  # Normalize to preserve DC gain
    
    # Upsample bits and apply pulse shaping filter
    bits_upsampled = jnp.repeat(bits, samples_per_symbol)
    expected_len = len(bits) * samples_per_symbol
    
    freq = jnp.convolve(bits_upsampled, g, mode='same')
    # Ensure output length matches expected (JAX convolve may add extra elements)
    freq = freq[:expected_len]
    
    # Integrate frequency deviation to get instantaneous phase (FM modulation)
    phase = jnp.cumsum(freq * jnp.pi * h)
    
    # Generate complex baseband signal with constant envelope
    return jnp.exp(1j * phase)


def generate_image(
    signal: np.ndarray,
    image_size: Tuple[int, int],
    image_dir: str,
    image_name: str
) -> None:
    """Generate constellation image from a complex signal.
    
    Args:
        signal: Complex signal array.
        image_size: Output image size (width, height).
        image_dir: Directory to save the image.
        image_name: Name for the output image file.
    """
    # Configuration parameters
    blk_size = [5, 25, 50]
    c_factor = 5.0 / np.array(blk_size)
    cons_scale = [2.5, 2.5]
    
    max_blk_size = max(blk_size)
    image_size_x = image_size[0] + 4 * max_blk_size
    image_size_y = image_size[1] + 4 * max_blk_size
    
    cons_scale_i = cons_scale[0] + 2 * max_blk_size * (2 * cons_scale[0] / image_size[0])
    cons_scale_q = cons_scale[1] + 2 * max_blk_size * (2 * cons_scale[1] / image_size[1])
    
    d_iy = 2 * cons_scale[0] / image_size[0]
    d_qx = 2 * cons_scale[1] / image_size[1]
    d_xy = np.sqrt(d_iy**2 + d_qx**2)
    
    # Calculate sample positions
    sample_x = np.rint((cons_scale_q - np.imag(signal)) / d_qx).astype(int)
    sample_y = np.rint((cons_scale_i + np.real(signal)) / d_iy).astype(int)
    
    # Create pixel centroid grid
    ii, jj = np.meshgrid(range(image_size_x), range(image_size_y), indexing='ij')
    pixel_centroid = (-cons_scale_i + d_iy / 2 + jj * d_iy) + 1j * (cons_scale_q - d_qx / 2 - ii * d_qx)
    
    image_array = np.zeros((image_size_x, image_size_y, 3))
    
    for kk, blk in enumerate(blk_size):
        blk_xmin = sample_x - blk
        blk_xmax = sample_x + blk + 1
        blk_ymin = sample_y - blk
        blk_ymax = sample_y + blk + 1
        
        valid = (blk_xmin > 0) & (blk_ymin > 0) & (blk_xmax < image_size_x) & (blk_ymax < image_size_y)
        
        for idx in np.where(valid)[0]:
            sample_distance = np.abs(signal[idx] - pixel_centroid[blk_xmin[idx]:blk_xmax[idx], 
                                                                   blk_ymin[idx]:blk_ymax[idx]])
            image_array[blk_xmin[idx]:blk_xmax[idx], blk_ymin[idx]:blk_ymax[idx], kk] += \
                np.exp(-c_factor[kk] * sample_distance / d_xy)
        
        max_val = np.max(image_array[:, :, kk])
        if max_val > 1e-8:  # Use epsilon for floating-point comparison
            image_array[:, :, kk] /= max_val
    
    # Convert to uint8 and save
    image_array = (image_array * 255).astype(np.uint8)
    im = Image.fromarray(image_array[2*max_blk_size:-2*max_blk_size, 2*max_blk_size:-2*max_blk_size])
    
    os.makedirs(image_dir, exist_ok=True)
    im.save(os.path.join(image_dir, f"{image_name}.png"))


def generate_constellation_images(
    mod_type: str,
    samples_per_image: int,
    image_num: int,
    image_size: Tuple[int, int],
    set_types: List[str],
    set_path: str,
    snr_range: Tuple[float, float] = (-10, 10),
    seed: int = 42
) -> None:
    """Generate constellation images for a given modulation type.
    
    Args:
        mod_type: Modulation type (e.g., 'QPSK', 'BPSK', 'GMSK').
        samples_per_image: Number of symbols per image.
        image_num: Number of images to generate.
        image_size: Output image size (width, height).
        set_types: Types of data to generate ('noiseLessImg', 'noisyImg', 
                   'noiselessSignal', 'noise', 'noisySignal').
        set_path: Base path for saving generated data.
        snr_range: SNR range in dB (min, max).
        seed: Random seed for reproducibility.
    """
    if mod_type not in MOD_TYPES:
        raise ValueError(f'Unrecognized modulation type: {mod_type}. '
                        f'Available types: {list(MOD_TYPES.keys())}')
    
    cons_diag, mod_order = MOD_TYPES[mod_type]
    
    # GMSK specific parameters
    bt_product = 0.3 if mod_type == 'GMSK' else None
    samples_per_symbol = 8 if mod_type == 'GMSK' else None
    
    # Create directories for each set type
    image_dirs = {gen_type: os.path.join(set_path, gen_type) for gen_type in set_types}
    for dir_path in image_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Set random seed
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    
    ll, ul = snr_range
    
    for jj in range(image_num):
        # Generate signal
        if mod_type == 'GMSK':
            msg_bits = np.random.randint(0, 2, samples_per_image)
            # Use JAX for GMSK modulation
            signal_tx = np.array(gmsk_modulate(jnp.array(msg_bits), bt_product, samples_per_symbol))
        else:
            msg = np.random.randint(len(cons_diag), size=samples_per_image)
            signal_tx = cons_diag[msg]
        
        signal_tx = signal_tx.astype(np.complex128)
        
        # Handle edge cases for visualization
        if mod_type in ['BPSK', '4ASK']:
            signal_tx[0] += 1j * 1E-4
        
        # Random SNR within range
        snr_dB = np.random.uniform(ll, ul)
        image_id_prefix = f"{mod_type}_{snr_dB:.2f}dB__"
        
        image_id = f"{jj:0{len(str(image_num))}d}"
        image_name = f"{image_id_prefix}{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Apply phase offset
        phase_offset = np.random.normal(0, 0.0001, len(signal_tx)) + \
                      np.arange(len(signal_tx)) * np.random.normal(0, 0.0001)
        signal_tx *= np.exp(1j * phase_offset)
        
        # Add noise using JAX
        key, subkey = jax.random.split(key)
        signal_rx, noise = awgn(jnp.array(signal_tx), snr_dB, subkey)
        signal_rx = np.array(signal_rx)
        noise = np.array(noise)
        
        # Generate outputs for each set type
        for gen_type in set_types:
            if gen_type == 'noiseLessImg':
                generate_image(
                    signal=signal_tx,
                    image_size=image_size,
                    image_dir=image_dirs[gen_type],
                    image_name=image_name
                )
            elif gen_type == 'noisyImg':
                generate_image(
                    signal=signal_rx,
                    image_size=image_size,
                    image_dir=image_dirs[gen_type],
                    image_name=image_name
                )
            elif gen_type == 'noiselessSignal':
                np.save(os.path.join(image_dirs[gen_type], f"{image_name}.npy"), signal_tx)
            elif gen_type == 'noise':
                np.save(os.path.join(image_dirs[gen_type], f"{image_name}.npy"), noise)
            elif gen_type == 'noisySignal':
                np.save(os.path.join(image_dirs[gen_type], f"{image_name}.npy"), signal_rx)
        
        # Log generated files
        with open(os.path.join(set_path, "files.txt"), 'a') as file:
            file.write(f"{image_name}\n")


def generate_batch_constellation(
    mod_types: List[str],
    samples_per_image: int,
    images_per_mod: int,
    image_size: Tuple[int, int],
    output_path: str,
    set_types: List[str],
    snr_range: Tuple[float, float] = (-10, 10),
    seed: int = 42
) -> None:
    """Generate constellation images for multiple modulation types.
    
    Args:
        mod_types: List of modulation types to generate.
        samples_per_image: Number of symbols per image.
        images_per_mod: Number of images per modulation type.
        image_size: Output image size (width, height).
        output_path: Base output directory.
        set_types: Types of data to generate.
        snr_range: SNR range in dB.
        seed: Random seed.
    """
    for i, mod_type in enumerate(mod_types):
        print(f"Generating {mod_type} ({i+1}/{len(mod_types)})...")
        generate_constellation_images(
            mod_type=mod_type,
            samples_per_image=samples_per_image,
            image_num=images_per_mod,
            image_size=image_size,
            set_types=set_types,
            set_path=output_path,
            snr_range=snr_range,
            seed=seed + i
        )
    print("Generation complete!")
