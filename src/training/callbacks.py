"""
Training Callbacks
Callbacks for early stopping, checkpointing, and learning rate scheduling.
"""

import torch
import numpy as np
import os
from typing import Optional


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors validation loss and stops training if no improvement for N epochs.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        # Set comparison function
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = np.inf
        else:  # mode == 'max'
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = -np.inf

    def __call__(self, current_score: float, epoch: int) -> bool:
        """
        Check if training should stop.

        Args:
            current_score: Current validation metric
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(current_score, self.best_score):
            # Improvement
            if self.verbose:
                print(
                    f"  Validation improved from {self.best_score:.6f} to {current_score:.6f}"
                )

            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            # No improvement
            self.counter += 1

            if self.verbose and self.counter > 0:
                print(
                    f"  No improvement for {self.counter} epoch(s) "
                    f"(patience: {self.patience})"
                )

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  Early stopping triggered after {epoch+1} epochs")
                    print(
                        f"  Best score: {self.best_score:.6f} at epoch {self.best_epoch+1}"
                    )
                return True

        return False


class ModelCheckpoint:
    """
    Save model checkpoints during training.

    Saves best model based on validation metric, and optionally saves periodic checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        metric_name: str = "loss",
        mode: str = "min",
        save_every: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            metric_name: Name of metric to monitor
            mode: 'min' or 'max'
            save_every: Save checkpoint every N epochs (None = only best)
            verbose: Print messages
        """
        self.checkpoint_dir = checkpoint_dir
        self.metric_name = metric_name
        self.mode = mode
        self.save_every = save_every
        self.verbose = verbose

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Set comparison function
        if mode == "min":
            self.is_better = lambda current, best: current < best
            self.best_score = np.inf
        else:
            self.is_better = lambda current, best: current > best
            self.best_score = -np.inf

        self.best_epoch = 0

    def __call__(
        self,
        model,
        optimizer,
        epoch: int,
        current_score: float,
        additional_info: Optional[dict] = None,
    ):
        """
        Save checkpoint if needed.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            current_score: Current validation metric
            additional_info: Additional info to save (scalers, config, etc.)
        """
        # Check if this is the best model
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch

            # Save best model
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            self._save_checkpoint(
                best_path, model, optimizer, epoch, current_score, additional_info
            )

            if self.verbose:
                print(f"  Saved best model to {best_path}")

        # Periodic checkpoint
        if self.save_every is not None and (epoch + 1) % self.save_every == 0:
            periodic_path = os.path.join(
                self.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
            )
            self._save_checkpoint(
                periodic_path, model, optimizer, epoch, current_score, additional_info
            )

            if self.verbose:
                print(f"  Saved periodic checkpoint to {periodic_path}")

    def _save_checkpoint(
        self,
        path: str,
        model,
        optimizer,
        epoch: int,
        score: float,
        additional_info: Optional[dict] = None,
    ):
        """Save checkpoint to file."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            f"{self.metric_name}": score,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)

    def load_best_model(self, model, optimizer=None):
        """
        Load the best saved model.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into

        Returns:
            Loaded checkpoint dict
        """
        best_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        if not os.path.exists(best_path):
            raise FileNotFoundError(f"No best model found at {best_path}")

        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.verbose:
            print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
            print(f"  Best {self.metric_name}: {checkpoint['best_score']:.6f}")

        return checkpoint


class LearningRateScheduler:
    """
    Reduce learning rate when validation metric plateaus.
    """

    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-7,
        verbose: bool = True,
    ):
        """
        Initialize LR scheduler.

        Args:
            optimizer: Optimizer to adjust
            mode: 'min' or 'max'
            factor: Factor to reduce LR by
            patience: Epochs to wait before reducing
            min_lr: Minimum learning rate
            verbose: Print messages
        """
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.counter = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        self.is_better = (
            (lambda current, best: current < best)
            if mode == "min"
            else (lambda current, best: current > best)
        )

    def __call__(self, current_score: float, epoch: int):
        """
        Potentially reduce learning rate.

        Args:
            current_score: Current validation metric
            epoch: Current epoch
        """
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self._reduce_lr()
                self.counter = 0

    def _reduce_lr(self):
        """Reduce learning rate."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = max(old_lr * self.factor, self.min_lr)

            if new_lr < old_lr:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"\n  Reducing learning rate: {old_lr:.2e} -> {new_lr:.2e}\n")


if __name__ == "__main__":
    # Test callbacks
    print("Training Callbacks")
    print("=" * 60)

    # Test Early Stopping
    print("\nTesting Early Stopping:")
    early_stop = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    val_losses = [1.0, 0.95, 0.92, 0.91, 0.905, 0.904, 0.903]

    for epoch, loss in enumerate(val_losses):
        print(f"Epoch {epoch+1}, Val Loss: {loss:.3f}")
        should_stop = early_stop(loss, epoch)
        if should_stop:
            print("Training would stop here!")
            break

    # Test Model Checkpoint
    print("\n\nTesting Model Checkpoint:")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(
            checkpoint_dir=tmpdir, metric_name="val_loss", mode="min", save_every=2
        )

        # Create dummy model
        dummy_model = torch.nn.Linear(10, 1)
        dummy_optimizer = torch.optim.Adam(dummy_model.parameters())

        for epoch, loss in enumerate([1.0, 0.8, 0.9, 0.7, 0.75]):
            print(f"\nEpoch {epoch+1}, Val Loss: {loss:.3f}")
            checkpoint(dummy_model, dummy_optimizer, epoch, loss)

        # Test loading
        print("\nLoading best model:")
        checkpoint.load_best_model(dummy_model, dummy_optimizer)

    print("\nCallbacks working correctly!")
