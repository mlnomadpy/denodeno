"""Evaluation utilities for DenoMAE using JAX/Flax.

This module provides evaluation functions for both reconstruction
and classification tasks.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

from src.denomae.model import DenoMAE, FineTunedDenoMAE
from src.denomae.mesh_utils import DataParallelTrainer


def evaluate_model(
    model: DenoMAE,
    dataloader,
    patch_size: int,
    image_size: int,
    key: jax.random.PRNGKey,
    trainer: Optional[DataParallelTrainer] = None
) -> Dict[str, float]:
    """Evaluate DenoMAE reconstruction performance.
    
    Args:
        model: DenoMAE model.
        dataloader: Data loader iterator.
        patch_size: Patch size.
        image_size: Image size.
        key: Random key.
        trainer: Optional data parallel trainer.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    from src.training.trainer import eval_step
    
    all_metrics = []
    
    for inputs, targets in dataloader:
        key, subkey = jax.random.split(key)
        
        if trainer is not None:
            inputs = [trainer.shard_batch(x) for x in inputs]
            targets = [trainer.shard_batch(x) for x in targets]
        
        _, metrics = eval_step(
            model, inputs, targets, patch_size, image_size, subkey
        )
        all_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {
        k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
        for k in all_metrics[0].keys()
    }
    
    return avg_metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        num_classes: Number of classes.
        
    Returns:
        Dictionary of metrics.
    """
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Per-class accuracy
    per_class_acc = []
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc.append(np.mean(y_pred[mask] == c))
    
    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        confusion_matrix[t, p] += 1
    
    return {
        'accuracy': float(accuracy),
        'mean_class_accuracy': float(np.mean(per_class_acc)),
        'confusion_matrix': confusion_matrix
    }


@jax.jit
def classification_forward(
    model: FineTunedDenoMAE,
    x: jnp.ndarray,
    deterministic: bool = True
) -> jnp.ndarray:
    """Forward pass for classification.
    
    Args:
        model: Fine-tuned model.
        x: Input tensor.
        deterministic: Whether to use deterministic mode.
        
    Returns:
        Logits.
    """
    return model(x, deterministic=deterministic)


def evaluate_classification(
    model: FineTunedDenoMAE,
    dataloader,
    num_classes: int,
    trainer: Optional[DataParallelTrainer] = None
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate classification model.
    
    Args:
        model: Fine-tuned DenoMAE model.
        dataloader: Data loader yielding (images, labels).
        num_classes: Number of classes.
        trainer: Optional data parallel trainer.
        
    Returns:
        Tuple of (metrics, all_preds, all_labels).
    """
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        if trainer is not None:
            images = trainer.shard_batch(images)
        
        logits = classification_forward(model, images)
        preds = jnp.argmax(logits, axis=-1)
        
        all_preds.extend(np.array(preds).tolist())
        all_labels.extend(np.array(labels).tolist())
        
        correct += int(jnp.sum(preds == labels))
        total += len(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = compute_classification_metrics(all_labels, all_preds, num_classes)
    metrics['total_accuracy'] = correct / total
    
    return metrics, all_preds, all_labels


def compute_snr_breakdown(
    model: FineTunedDenoMAE,
    dataloader,
    snr_values: List[float],
    num_classes: int
) -> Dict[float, Dict[str, float]]:
    """Compute accuracy breakdown by SNR.
    
    Args:
        model: Fine-tuned model.
        dataloader: Data loader yielding (images, labels, snr).
        snr_values: List of SNR values to evaluate.
        num_classes: Number of classes.
        
    Returns:
        Dictionary mapping SNR to metrics.
    """
    snr_metrics = {}
    
    for snr in snr_values:
        snr_preds = []
        snr_labels = []
        
        for images, labels, snr_batch in dataloader:
            mask = snr_batch == snr
            if not jnp.any(mask):
                continue
            
            masked_images = images[mask]
            masked_labels = labels[mask]
            
            logits = classification_forward(model, masked_images)
            preds = jnp.argmax(logits, axis=-1)
            
            snr_preds.extend(np.array(preds).tolist())
            snr_labels.extend(np.array(masked_labels).tolist())
        
        if snr_preds:
            snr_preds = np.array(snr_preds)
            snr_labels = np.array(snr_labels)
            snr_metrics[snr] = compute_classification_metrics(
                snr_labels, snr_preds, num_classes
            )
    
    return snr_metrics


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = 'Confusion Matrix'
) -> None:
    """Plot and save confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
        save_path: Path to save the plot.
        title: Plot title.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(confusion_matrix[i, j], 'd'),
                ha='center', va='center',
                color='white' if confusion_matrix[i, j] > thresh else 'black'
            )
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")
