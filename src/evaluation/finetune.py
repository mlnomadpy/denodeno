"""Fine-tuning script for DenoMAE classification using Flax NNX.

This script handles fine-tuning of pretrained DenoMAE for classification.
"""

import os
import argparse
from typing import Optional, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from src.denomae.model import DenoMAE, FineTunedDenoMAE
from src.denomae.mesh_utils import create_device_mesh, DataParallelTrainer
from src.data.datagen import ImageFolderDataset, create_classification_loader
from src.training.trainer import load_checkpoint
from src.training.wandb_utils import WandBLogger, log_metrics
from src.evaluation.evaluator import (
    evaluate_classification,
    compute_classification_metrics,
    plot_confusion_matrix,
)


class FineTuner:
    """Fine-tuning manager for DenoMAE classification.
    
    Args:
        pretrained_model: Pretrained DenoMAE model.
        num_classes: Number of output classes.
        learning_rate: Learning rate.
        freeze_encoder: Whether to freeze encoder weights.
    """
    
    def __init__(
        self,
        pretrained_model: DenoMAE,
        num_classes: int,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
        rngs: nnx.Rngs = None
    ):
        self.model = FineTunedDenoMAE(
            backbone=pretrained_model,
            num_classes=num_classes,
            freeze_encoder=freeze_encoder,
            rngs=rngs
        )
        
        # Create optimizer with wrt=nnx.Param
        self.optimizer = nnx.Optimizer(
            self.model,
            optax.adam(learning_rate=learning_rate),
            wrt=nnx.Param
        )
    
    def _train_step_impl(
        self,
        images: jnp.ndarray,
        labels: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Execute a single training step.
        
        Args:
            images: Input images.
            labels: Ground truth labels.
            
        Returns:
            Metrics dict.
        """
        model = self.model
        
        @nnx.jit
        def compute_and_update(model, optimizer, images, labels):
            def loss_fn(model):
                logits = model(images, deterministic=False)
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                return jnp.mean(loss), logits
            
            (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            
            # Compute accuracy
            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == labels)
            
            return loss, accuracy
        
        loss, accuracy = compute_and_update(model, self.optimizer, images, labels)
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def train_epoch(
        self,
        dataloader,
        trainer: Optional[DataParallelTrainer] = None
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Data loader.
            trainer: Optional data parallel trainer.
            
        Returns:
            Epoch metrics.
        """
        all_metrics = []
        
        for images, labels in dataloader:
            if trainer is not None:
                images = trainer.shard_batch(images)
                labels = trainer.shard_batch(labels)
            
            metrics = self._train_step_impl(images, labels)
            all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_metrics = {
            k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
            for k in all_metrics[0].keys()
        }
        
        return avg_metrics
    
    def evaluate(
        self,
        dataloader,
        num_classes: int,
        trainer: Optional[DataParallelTrainer] = None
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate the model.
        
        Args:
            dataloader: Data loader.
            num_classes: Number of classes.
            trainer: Optional data parallel trainer.
            
        Returns:
            Tuple of (metrics, predictions, labels).
        """
        return evaluate_classification(
            self.model, dataloader, num_classes, trainer
        )
    
    def save(self, path: str) -> None:
        """Save model state.
        
        Args:
            path: Path to save.
        """
        import pickle
        
        model_state = nnx.state(self.model)
        
        with open(path, 'wb') as f:
            pickle.dump({'model_state': model_state}, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model state.
        
        Args:
            path: Path to load from.
        """
        import pickle
        
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        nnx.update(self.model, checkpoint['model_state'])
        print(f"Model loaded from {path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fine-tune DenoMAE for Classification')
    
    # Model arguments
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--in_chans', type=int, default=3, help='Input channels')
    parser.add_argument('--mask_ratio', type=float, default=0.0, help='Mask ratio')
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--encoder_depth', type=int, default=12, help='Encoder depth')
    parser.add_argument('--decoder_depth', type=int, default=4, help='Decoder depth')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--num_modality', type=int, default=1, help='Number of modalities')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder')
    
    # Path arguments
    parser.add_argument('--pretrained_model_path', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_model_path', type=str, default='models/finetuned_denomae.pkl',
                        help='Path to save model')
    parser.add_argument('--confusion_matrix_path', type=str, default='results/confusion_matrix.png',
                        help='Path to save confusion matrix')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='denomae-finetune',
                        help='WandB project')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
    
    # TPU arguments
    parser.add_argument('--use_tpu', action='store_true', help='Use TPU')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def finetune_main():
    """Main fine-tuning function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.confusion_matrix_path), exist_ok=True)
    
    # Print device info
    print(f"JAX devices: {jax.devices()}")
    print(f"Number of devices: {jax.device_count()}")
    
    # Setup mesh for TPU/multi-device
    trainer = None
    if args.use_tpu or jax.device_count() > 1:
        mesh = create_device_mesh(axis_names=('data',))
        trainer = DataParallelTrainer(mesh)
    
    # Initialize random key
    key = jax.random.PRNGKey(args.seed)
    key, model_key = jax.random.split(key)
    rngs = nnx.Rngs(model_key)
    
    # Create pretrained model architecture
    pretrained_model = DenoMAE(
        num_modalities=args.num_modality,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        mask_ratio=args.mask_ratio,
        embed_dim=args.embed_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        rngs=rngs
    )
    
    # Load pretrained weights
    pretrained_model, _ = load_checkpoint(pretrained_model, args.pretrained_model_path)
    print(f"Loaded pretrained model from {args.pretrained_model_path}")
    
    # Create fine-tuner
    key, finetune_key = jax.random.split(key)
    finetune_rngs = nnx.Rngs(finetune_key)
    
    finetuner = FineTuner(
        pretrained_model=pretrained_model,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        freeze_encoder=args.freeze_encoder,
        rngs=finetune_rngs
    )
    
    # Create datasets
    train_dataset = ImageFolderDataset(
        root=os.path.join(args.data_path, 'train'),
        image_size=(args.img_size, args.img_size)
    )
    test_dataset = ImageFolderDataset(
        root=os.path.join(args.data_path, 'test'),
        image_size=(args.img_size, args.img_size)
    )
    
    print(f"Classes: {train_dataset.classes}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Initialize WandB
    config = vars(args)
    
    with WandBLogger(
        project=args.wandb_project,
        name=f"finetune_{args.num_classes}classes",
        config=config,
        entity=args.wandb_entity,
        tags=['denomae', 'finetune', 'jax']
    ) as logger:
        
        best_accuracy = 0.0
        
        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            
            # Get data loaders
            train_loader = create_classification_loader(
                train_dataset, args.batch_size, shuffle=True, seed=args.seed + epoch
            )
            test_loader = create_classification_loader(
                test_dataset, args.batch_size, shuffle=False
            )
            
            # Train
            train_metrics = finetuner.train_epoch(train_loader, trainer)
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']*100:.2f}%")
            logger.log(train_metrics, prefix='train')
            
            # Evaluate every 5 epochs
            if (epoch + 1) % 5 == 0:
                test_loader = create_classification_loader(
                    test_dataset, args.batch_size, shuffle=False
                )
                metrics, preds, labels = finetuner.evaluate(
                    test_loader, args.num_classes, trainer
                )
                
                print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%")
                logger.log(metrics, prefix='test')
                
                # Save best model
                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    finetuner.save(args.save_model_path)
                    print(f"New best model saved with accuracy: {best_accuracy*100:.2f}%")
                    
                    # Save confusion matrix
                    plot_confusion_matrix(
                        metrics['confusion_matrix'],
                        train_dataset.classes,
                        args.confusion_matrix_path,
                        title=f'Confusion Matrix - Accuracy: {best_accuracy*100:.2f}%'
                    )
                    
                    # Log to WandB
                    logger.log_confusion_matrix(
                        labels, preds, train_dataset.classes
                    )
        
        print(f"\nTraining completed. Best accuracy: {best_accuracy*100:.2f}%")
        
        # Final evaluation
        finetuner.load(args.save_model_path)
        test_loader = create_classification_loader(
            test_dataset, args.batch_size, shuffle=False
        )
        final_metrics, final_preds, final_labels = finetuner.evaluate(
            test_loader, args.num_classes, trainer
        )
        
        print(f"Final Test Accuracy: {final_metrics['accuracy']*100:.2f}%")


if __name__ == '__main__':
    finetune_main()
