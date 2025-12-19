"""
Base Time Series Model
Abstract base class for all time series prediction models.
"""

from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseTimeSeriesModel(ABC, nn.Module):
    """
    Abstract base class for all time series models.

    This provides a common interface for LSTM, Transformer, and statistical models,
    ensuring consistency across different model architectures.
    """

    def __init__(self, config: Any):
        """
        Initialize base model.

        Args:
            config: Model configuration object with hyperparameters
        """
        super().__init__()
        self.config = config
        self.input_size = getattr(config, "input_size", 1)
        self.output_size = getattr(config, "output_size", 1)

    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Model output (implementation-specific)
        """
        pass

    @abstractmethod
    def predict(self, x, horizon: int):
        """
        Make predictions for a specific time horizon.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            horizon: Number of timesteps to predict ahead

        Returns:
            Predictions of shape (batch_size, horizon)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata and configuration.

        Returns:
            Dictionary containing model architecture name, config, and parameter count
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": self.__class__.__name__,
            "config": (
                self.config.__dict__
                if hasattr(self.config, "__dict__")
                else str(self.config)
            ),
            "trainable_parameters": total_params,
            "input_size": self.input_size,
            "output_size": self.output_size,
        }

    def save_model(self, path: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save the model
            additional_info: Optional dict with scalers, metadata, etc.
        """
        import torch

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_info": self.get_model_info(),
            "config": self.config,
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, path)

    @classmethod
    def load_model(cls, path: str, map_location: str = "cpu"):
        """
        Load model from checkpoint.

        Args:
            path: Path to the saved model
            map_location: Device to load model to

        Returns:
            Loaded model and additional info
        """
        import torch

        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint["config"]

        # Instantiate model
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Return model and additional info
        additional_info = {
            k: v
            for k, v in checkpoint.items()
            if k not in ["model_state_dict", "model_info", "config"]
        }

        return model, additional_info


class StatisticalBaseModel(ABC):
    """
    Base class for statistical time series models (ARIMA, Prophet, etc.)
    These don't inherit from nn.Module since they're not neural networks.
    """

    def __init__(self, config: Any):
        """
        Initialize statistical model.

        Args:
            config: Model configuration
        """
        self.config = config
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, y_train, **kwargs):
        """
        Fit the statistical model on training data.

        Args:
            y_train: Training time series data
            **kwargs: Additional fitting parameters
        """
        pass

    @abstractmethod
    def predict(self, steps: int, **kwargs):
        """
        Make predictions for N steps ahead.

        Args:
            steps: Number of timesteps to predict
            **kwargs: Additional prediction parameters

        Returns:
            Array of predictions
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return model metadata.

        Returns:
            Dictionary with model information
        """
        return {
            "architecture": self.__class__.__name__,
            "config": (
                self.config if isinstance(self.config, dict) else str(self.config)
            ),
            "is_fitted": self.is_fitted,
        }

    def save_model(self, path: str):
        """
        Save statistical model.

        Args:
            path: Path to save the model
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "config": self.config,
                    "is_fitted": self.is_fitted,
                },
                f,
            )

    @classmethod
    def load_model(cls, path: str):
        """
        Load statistical model from file.

        Args:
            path: Path to saved model

        Returns:
            Loaded model instance
        """
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        model_instance = cls(data["config"])
        model_instance.model = data["model"]
        model_instance.is_fitted = data["is_fitted"]

        return model_instance
