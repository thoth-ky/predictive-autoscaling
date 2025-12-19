"""
Custom Loss Functions for Time Series Models
"""

import torch
import torch.nn as nn
from typing import Dict


class MultiHorizonLoss(nn.Module):
    """
    Loss function for multi-horizon prediction.

    Combines losses from multiple prediction horizons with configurable weights.
    Allows different importance for different horizons (e.g., prioritize 15-min over 30-min).
    """

    def __init__(
        self, horizon_weights: Dict[int, float] = None, base_loss: str = "mse"
    ):
        """
        Initialize multi-horizon loss.

        Args:
            horizon_weights: Dict mapping horizon to weight
                           e.g., {20: 1.0, 60: 1.5, 120: 1.0}
                           If None, all horizons weighted equally
            base_loss: Base loss function ('mse', 'mae', 'huber')
        """
        super().__init__()

        self.horizon_weights = horizon_weights or {}
        self.base_loss = base_loss

        # Select base loss function
        if base_loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif base_loss == "mae":
            self.loss_fn = nn.L1Loss()
        elif base_loss == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss function: {base_loss}")

    def forward(
        self, predictions: Dict[int, torch.Tensor], targets: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate weighted loss across all horizons.

        Args:
            predictions: Dict mapping horizon to predicted tensors
                        {20: (batch, 20), 60: (batch, 60), ...}
            targets: Dict mapping horizon to target tensors

        Returns:
            Weighted combined loss
        """
        total_loss = 0.0
        total_weight = 0.0

        for horizon in predictions.keys():
            pred = predictions[horizon]
            target = targets[horizon]

            # Calculate loss for this horizon
            loss_h = self.loss_fn(pred, target)

            # Apply weight
            weight = self.horizon_weights.get(horizon, 1.0)
            total_loss += weight * loss_h
            total_weight += weight

        # Normalize by total weight
        if total_weight > 0:
            total_loss = total_loss / total_weight

        return total_loss


class WeightedMSELoss(nn.Module):
    """
    MSE loss with time-decaying weights.

    Gives higher importance to near-term predictions within a horizon.
    """

    def __init__(self, decay_rate: float = 0.95):
        """
        Initialize weighted MSE loss.

        Args:
            decay_rate: Exponential decay rate for weights (0-1)
                       1.0 = no decay (equal weights)
                       <1.0 = decay (near-term more important)
        """
        super().__init__()
        self.decay_rate = decay_rate

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted MSE.

        Args:
            pred: Predictions of shape (batch, horizon)
            target: Targets of shape (batch, horizon)

        Returns:
            Weighted MSE loss
        """
        horizon = pred.shape[1]

        # Create exponentially decaying weights
        weights = torch.pow(self.decay_rate, torch.arange(horizon, dtype=torch.float32))
        weights = weights.to(pred.device)

        # Normalize weights to sum to horizon (for comparability with standard MSE)
        weights = weights * (horizon / weights.sum())

        # Reshape for broadcasting
        weights = weights.view(1, -1)

        # Weighted squared error
        squared_error = (pred - target) ** 2
        weighted_error = squared_error * weights

        return weighted_error.mean()


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for the loss

    Returns:
        Loss function module
    """
    if loss_name == "mse":
        return nn.MSELoss(**kwargs)
    elif loss_name == "mae" or loss_name == "l1":
        return nn.L1Loss(**kwargs)
    elif loss_name == "huber":
        return nn.HuberLoss(**kwargs)
    elif loss_name == "weighted_mse":
        return WeightedMSELoss(**kwargs)
    elif loss_name == "multi_horizon":
        return MultiHorizonLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    print("Custom Loss Functions")
    print("=" * 60)

    batch_size = 32
    horizons = [20, 60, 120]

    # Create dummy predictions and targets
    predictions = {h: torch.randn(batch_size, h) for h in horizons}
    targets = {h: torch.randn(batch_size, h) for h in horizons}

    # Test multi-horizon loss
    print("\nMulti-Horizon Loss:")
    weights = {20: 1.0, 60: 1.5, 120: 1.0}
    multi_loss = MultiHorizonLoss(horizon_weights=weights, base_loss="mse")
    loss_value = multi_loss(predictions, targets)
    print(f"  Loss value: {loss_value.item():.4f}")
    print(f"  Horizon weights: {weights}")

    # Test weighted MSE
    print("\nWeighted MSE Loss:")
    weighted_mse = WeightedMSELoss(decay_rate=0.95)
    pred_single = predictions[60]
    target_single = targets[60]
    loss_value = weighted_mse(pred_single, target_single)
    print(f"  Loss value: {loss_value.item():.4f}")

    # Compare with standard MSE
    standard_mse = nn.MSELoss()
    standard_loss = standard_mse(pred_single, target_single)
    print(f"  Standard MSE: {standard_loss.item():.4f}")

    print("\nLoss functions working correctly!")
