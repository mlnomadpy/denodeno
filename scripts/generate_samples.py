#!/usr/bin/env python
"""Script to generate sample data for training.

This script generates constellation images for various modulation types
for both training and testing.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.sample_generation import generate_constellation_images


# Configuration
SAMPLES_PER_IMAGE = 1024
IMAGE_SIZE = (224, 224)
IMAGE_NUM = 100
MOD_TYPES = ['OOK', '4ASK', '8ASK', 'OQPSK', 'CPFSK', 'GFSK', '4PAM', 'DQPSK', '16PAM', 'GMSK']
SET_TYPES = ['noiseLessImg', 'noisyImg', 'noiselessSignal', 'noise', 'noisySignal']
BASE_PATH = './data'
SNR_RANGE = (-10, 10)


def main():
    """Generate training and test data."""
    # Generate training data
    print('Generating training data...')
    train_path = os.path.join(BASE_PATH, 'unlabeled', 'train')
    os.makedirs(train_path, exist_ok=True)

    for mod_type in MOD_TYPES:
        print(f'  Generating {mod_type}...')
        generate_constellation_images(
            mod_type=mod_type,
            samples_per_image=SAMPLES_PER_IMAGE,
            image_num=IMAGE_NUM,
            image_size=IMAGE_SIZE,
            set_types=SET_TYPES,
            set_path=train_path,
            snr_range=SNR_RANGE,
            seed=42
        )

    # Generate test data
    print('Generating test data...')
    test_path = os.path.join(BASE_PATH, 'unlabeled', 'test')
    os.makedirs(test_path, exist_ok=True)

    for mod_type in MOD_TYPES:
        print(f'  Generating {mod_type}...')
        generate_constellation_images(
            mod_type=mod_type,
            samples_per_image=SAMPLES_PER_IMAGE,
            image_num=IMAGE_NUM // 5,  # Fewer test samples
            image_size=IMAGE_SIZE,
            set_types=SET_TYPES,
            set_path=test_path,
            snr_range=SNR_RANGE,
            seed=123
        )

    print('Data generation complete!')


if __name__ == '__main__':
    main()
