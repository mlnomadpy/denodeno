"""Weights & Biases integration for training and evaluation logging.

This module provides utilities for logging metrics, images, and artifacts
to Weights & Biases (wandb) during training and evaluation.
"""

import os
from typing import Dict, Any, Optional, List, Union
import numpy as np


def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    resume: bool = False,
    run_id: Optional[str] = None,
    entity: Optional[str] = None
) -> Any:
    """Initialize a Weights & Biases run.
    
    Args:
        project: WandB project name.
        name: Run name.
        config: Configuration dictionary.
        tags: List of tags for the run.
        notes: Notes for the run.
        resume: Whether to resume a previous run.
        run_id: ID of run to resume.
        entity: WandB entity (team or username).
        
    Returns:
        WandB run object.
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("Please install wandb: pip install wandb")
    
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags,
        notes=notes,
        resume='must' if resume else 'allow',
        id=run_id,
        entity=entity
    )
    
    return run


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ''
) -> None:
    """Log metrics to WandB.
    
    Args:
        metrics: Dictionary of metric names to values.
        step: Global step number.
        prefix: Prefix to add to metric names.
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    wandb.log(metrics, step=step)


def log_images(
    images: Dict[str, np.ndarray],
    step: Optional[int] = None,
    caption: Optional[str] = None
) -> None:
    """Log images to WandB.
    
    Args:
        images: Dictionary of image names to numpy arrays.
                Arrays should be (H, W, C) or (H, W) for grayscale.
        step: Global step number.
        caption: Caption for the images.
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    wandb_images = {}
    for name, img in images.items():
        # Handle different image formats
        if len(img.shape) == 4:  # Batch of images
            img = img[0]  # Take first image
        
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            # Channels-first format (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0, 1] if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        # Clip values
        img = np.clip(img, 0, 1)
        
        wandb_images[name] = wandb.Image(img, caption=caption)
    
    wandb.log(wandb_images, step=step)


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    step: Optional[int] = None,
    title: str = 'Confusion Matrix'
) -> None:
    """Log a confusion matrix to WandB.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        step: Global step number.
        title: Title for the confusion matrix.
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    wandb.log({
        title: wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=class_names
        )
    }, step=step)


def log_histogram(
    name: str,
    values: np.ndarray,
    step: Optional[int] = None
) -> None:
    """Log a histogram to WandB.
    
    Args:
        name: Name of the histogram.
        values: Values to create histogram from.
        step: Global step number.
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    wandb.log({name: wandb.Histogram(values)}, step=step)


def log_model(
    model_path: str,
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Log a model artifact to WandB.
    
    Args:
        model_path: Path to the model file.
        name: Name for the artifact.
        metadata: Additional metadata.
    """
    try:
        import wandb
    except ImportError:
        return
    
    if wandb.run is None:
        return
    
    artifact = wandb.Artifact(
        name=name,
        type='model',
        metadata=metadata
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)


def finish_wandb() -> None:
    """Finish the current WandB run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass


class WandBLogger:
    """Context manager for WandB logging.
    
    Args:
        project: WandB project name.
        name: Run name.
        config: Configuration dictionary.
        tags: List of tags.
        notes: Notes for the run.
        entity: WandB entity.
    """
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        entity: Optional[str] = None
    ):
        self.project = project
        self.name = name
        self.config = config
        self.tags = tags
        self.notes = notes
        self.entity = entity
        self.run = None
        self._step = 0
    
    def __enter__(self):
        self.run = init_wandb(
            project=self.project,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            entity=self.entity
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        finish_wandb()
        return False
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ''
    ) -> None:
        """Log metrics."""
        if step is None:
            step = self._step
            self._step += 1
        log_metrics(metrics, step=step, prefix=prefix)
    
    def log_images(
        self,
        images: Dict[str, np.ndarray],
        step: Optional[int] = None,
        caption: Optional[str] = None
    ) -> None:
        """Log images."""
        if step is None:
            step = self._step
        log_images(images, step=step, caption=caption)
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        step: Optional[int] = None,
        title: str = 'Confusion Matrix'
    ) -> None:
        """Log confusion matrix."""
        if step is None:
            step = self._step
        log_confusion_matrix(y_true, y_pred, class_names, step=step, title=title)
    
    def save_model(
        self,
        model_path: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model as artifact."""
        log_model(model_path, name, metadata)


class TrainingMetrics:
    """Helper class to track and compute training metrics.
    
    Args:
        smooth_factor: Factor for exponential moving average.
    """
    
    def __init__(self, smooth_factor: float = 0.9):
        self.smooth_factor = smooth_factor
        self._metrics: Dict[str, List[float]] = {}
        self._smoothed: Dict[str, float] = {}
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics with new values."""
        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
                self._smoothed[name] = value
            
            self._metrics[name].append(value)
            self._smoothed[name] = (
                self.smooth_factor * self._smoothed[name] + 
                (1 - self.smooth_factor) * value
            )
    
    def get_smoothed(self) -> Dict[str, float]:
        """Get smoothed metrics."""
        return self._smoothed.copy()
    
    def get_average(self) -> Dict[str, float]:
        """Get average of all recorded metrics."""
        return {
            name: np.mean(values) 
            for name, values in self._metrics.items()
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._smoothed.clear()
