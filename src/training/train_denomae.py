"""Main training script for DenoMAE using Flax NNX and JAX.

This script handles training of the DenoMAE model with:
- TPU v5-8 support with pure data parallelism
- WandB logging
- HuggingFace data streaming support
"""

import os
import argparse
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.model import DenoMAE
from src.denomae.mesh_utils import create_device_mesh, DataParallelTrainer, print_mesh_info
from src.data.datagen import DenoMAEDataset, create_data_loader
from src.data.huggingface_utils import HuggingFaceDataLoader
from src.training.trainer import (
    create_train_state,
    train_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
)
from src.training.wandb_utils import WandBLogger, log_metrics, log_images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DenoMAE Training Script (JAX/Flax)")
    
    # Data arguments
    parser.add_argument("--train_path", type=str, default='./data/unlabeled/train/',
                        help="Path to training data")
    parser.add_argument("--test_path", type=str, default='./data/unlabeled/test/',
                        help="Path to test data")
    parser.add_argument("--hf_dataset", type=str, default=None,
                        help="HuggingFace dataset ID for streaming")
    
    # Model arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224], help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--in_chans", type=int, default=3, help="Input channels")
    parser.add_argument("--mask_ratio", type=float, default=0.75, help="Mask ratio")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Encoder depth")
    parser.add_argument("--decoder_depth", type=int, default=4, help="Decoder depth")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num_modality", type=int, default=5, help="Number of modalities")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    
    # Output arguments
    parser.add_argument("--model_dir", type=str, default='models', help="Model save directory")
    parser.add_argument("--model_name", type=str, default='denomae', help="Model name prefix")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default='denomae', help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--log_interval", type=int, default=50, help="Logging interval")
    
    # TPU arguments
    parser.add_argument("--use_tpu", action="store_true", help="Use TPU for training")
    parser.add_argument("--tpu_mesh_shape", type=int, nargs='+', default=[8],
                        help="TPU mesh shape for data parallelism")
    
    # Misc arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load", action="store_true", help="Load checkpoint")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Print device info
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Setup mesh for TPU/multi-device training
    trainer = None
    if args.use_tpu or jax.device_count() > 1:
        mesh = create_device_mesh(
            mesh_shape=tuple(args.tpu_mesh_shape),
            axis_names=('data',)
        )
        trainer = DataParallelTrainer(mesh)
        print("Device mesh setup:")
        print_mesh_info(mesh)
    
    # Initialize random key
    key = jax.random.PRNGKey(args.seed)
    
    # Create model
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)
    
    model = DenoMAE(
        num_modalities=args.num_modality,
        img_size=args.image_size[0],
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        mask_ratio=args.mask_ratio,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        rngs=rngs
    )
    
    print(f"Model created with {args.num_modality} modalities")
    
    # Create train state
    state = create_train_state(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if requested
    start_epoch = 0
    best_loss = float('inf')
    
    if args.load:
        checkpoint_path = os.path.join(
            args.model_dir, 
            f"{args.model_name}_{args.num_modality}.pkl"
        )
        if os.path.exists(checkpoint_path):
            model, metadata = load_checkpoint(model, checkpoint_path)
            start_epoch = metadata['epoch']
            best_loss = metadata['best_loss']
            print(f"Loaded checkpoint from epoch {start_epoch}")
    
    # Create data loaders
    if args.hf_dataset:
        # Use HuggingFace streaming
        print(f"Using HuggingFace dataset: {args.hf_dataset}")
        train_loader = HuggingFaceDataLoader(
            repo_id=args.hf_dataset,
            split='train',
            dataset_type='unlabeled',
            num_modalities=args.num_modality,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size
        )
        test_loader = HuggingFaceDataLoader(
            repo_id=args.hf_dataset,
            split='test',
            dataset_type='unlabeled',
            num_modalities=args.num_modality,
            image_size=tuple(args.image_size),
            batch_size=args.batch_size
        )
    else:
        # Use local data
        train_config = {
            'noisy_image_path': os.path.join(args.train_path, 'noisyImg/'),
            'noiseless_img_path': os.path.join(args.train_path, 'noiseLessImg/'),
            'noisy_signal_path': os.path.join(args.train_path, 'noisySignal/'),
            'noiseless_signal_path': os.path.join(args.train_path, 'noiselessSignal/'),
            'noise_path': os.path.join(args.train_path, 'noise/'),
        }
        
        test_config = {
            'noisy_image_path': os.path.join(args.test_path, 'noisyImg/'),
            'noiseless_img_path': os.path.join(args.test_path, 'noiseLessImg/'),
            'noisy_signal_path': os.path.join(args.test_path, 'noisySignal/'),
            'noiseless_signal_path': os.path.join(args.test_path, 'noiselessSignal/'),
            'noise_path': os.path.join(args.test_path, 'noise/'),
        }
        
        train_dataset = DenoMAEDataset(
            num_modalities=args.num_modality,
            image_size=tuple(args.image_size),
            **train_config
        )
        test_dataset = DenoMAEDataset(
            num_modalities=args.num_modality,
            image_size=tuple(args.image_size),
            **test_config
        )
        
        def get_train_loader():
            return create_data_loader(train_dataset, args.batch_size, shuffle=True)
        
        def get_test_loader():
            return create_data_loader(test_dataset, args.batch_size, shuffle=False)
    
    # Initialize WandB
    config = vars(args)
    
    with WandBLogger(
        project=args.wandb_project,
        name=f"{args.model_name}_{args.num_modality}mod",
        config=config,
        entity=args.wandb_entity,
        tags=['denomae', 'jax', 'flax-nnx']
    ) as logger:
        
        # Training loop
        for epoch in range(start_epoch, args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            
            # Get fresh iterators
            if args.hf_dataset:
                train_iter = iter(train_loader)
                test_iter = iter(test_loader)
            else:
                train_iter = get_train_loader()
                test_iter = get_test_loader()
            
            # Train for one epoch
            key, train_key = jax.random.split(key)
            state, train_metrics = train_epoch(
                state=state,
                dataloader=train_iter,
                patch_size=args.patch_size,
                image_size=args.image_size[0],
                key=train_key,
                trainer=trainer,
                log_interval=args.log_interval
            )
            
            print(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}")
            logger.log(train_metrics, prefix='train')
            
            # Evaluate
            key, eval_key = jax.random.split(key)
            reconstructions, test_metrics = evaluate(
                model=state.model,
                dataloader=test_iter,
                patch_size=args.patch_size,
                image_size=args.image_size[0],
                key=eval_key,
                trainer=trainer
            )
            
            print(f"Epoch {epoch + 1} - Test Loss: {test_metrics['loss']:.4f}")
            logger.log(test_metrics, prefix='test')
            
            # Log reconstructions
            if reconstructions is not None and len(reconstructions) > 0:
                images_to_log = {
                    f'reconstruction_{i}': jnp.array(rec[0])  # First sample
                    for i, rec in enumerate(reconstructions)
                }
                logger.log_images(images_to_log, step=epoch)
            
            # Save best model
            if test_metrics['loss'] < best_loss:
                best_loss = test_metrics['loss']
                checkpoint_path = os.path.join(
                    args.model_dir,
                    f"{args.model_name}_{args.num_modality}.pkl"
                )
                save_checkpoint(state, checkpoint_path, epoch + 1, best_loss)
                print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(
            args.model_dir,
            f"{args.model_name}_final_{args.num_modality}_{args.num_epochs}.pkl"
        )
        save_checkpoint(state, final_path, args.num_epochs, best_loss)
        
        print("\nTraining completed!")
        print(f"Best test loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
