"""
Configuration System
Dataclasses for model, data, and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Configuration for data processing and windowing."""

    raw_data_path: str = "/home/thoth/dpn/predictive-autoscaling/data/raw"
    processed_data_path: str = "/home/thoth/dpn/predictive-autoscaling/data/processed"

    # Window configuration
    window_size: int = 240  # 60 minutes at 15s intervals
    prediction_horizons: List[int] = field(
        default_factory=lambda: [20, 60, 120]
    )  # 5, 15, 30 min
    stride: int = 4  # 1 minute between windows

    # Train/val/test split ratios
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Feature engineering
    include_temporal_features: bool = True
    include_lag_features: bool = True
    include_rolling_features: bool = True

    # Normalization
    normalization: str = "minmax"  # Options: 'minmax', 'standard', 'robust'
    outlier_threshold: Optional[float] = 3.0  # Std deviations for outlier detection


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_type: str = "lstm"  # Options: 'lstm', 'arima', 'prophet'
    input_size: int = 1  # Number of features
    output_size: int = 1  # Always 1 for univariate prediction

    # LSTM-specific parameters
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Statistical model parameters (ARIMA)
    arima_order: tuple = (1, 1, 1)  # (p, d, q)
    arima_seasonal_order: tuple = (0, 0, 0, 0)  # (P, D, Q, s)

    # Prophet parameters
    prophet_changepoint_prior_scale: float = 0.05
    prophet_yearly_seasonality: bool = False
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"  # Options: 'adam', 'sgd', 'rmsprop'
    weight_decay: float = 0.0

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Loss function
    loss_function: str = "mse"  # Options: 'mse', 'mae', 'huber'

    # Multi-horizon loss weights
    horizon_weights: Dict[int, float] = field(
        default_factory=lambda: {
            20: 1.0,  # 5-min horizon
            60: 1.5,  # 15-min horizon (slightly higher weight)
            120: 1.0,  # 30-min horizon
        }
    )

    # Device
    device: str = "cpu"  # Will be set to 'cuda' if available
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = (
        "/home/thoth/dpn/predictive-autoscaling/experiments/checkpoints"
    )
    save_every_n_epochs: int = 10

    # Experiment tracking
    experiment_name: str = "predictive-autoscaling"
    tracking_uri: str = "/home/thoth/dpn/predictive-autoscaling/experiments/runs"
    log_interval: int = 10  # Log every N batches

    # SageMaker specific
    use_sagemaker: bool = False
    sagemaker_instance_type: str = "ml.p3.2xlarge"


@dataclass
class ExperimentConfig:
    """Complete configuration combining all components."""

    metric_name: str  # e.g., 'cpu', 'memory', 'disk_reads'
    container_name: str = "webapp"  # Target container

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "metric_name": self.metric_name,
            "container_name": self.container_name,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary."""
        data_config = DataConfig(**config_dict.get("data", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))

        return cls(
            metric_name=config_dict["metric_name"],
            container_name=config_dict.get("container_name", "webapp"),
            data=data_config,
            model=model_config,
            training=training_config,
        )

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def save_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig object
    """
    return ExperimentConfig.from_yaml(config_path)


def create_default_config(
    metric_name: str, model_type: str = "lstm"
) -> ExperimentConfig:
    """
    Create a default configuration for a specific metric.

    Args:
        metric_name: Name of metric (cpu, memory, disk_reads, etc.)
        model_type: Type of model (lstm, arima, prophet)

    Returns:
        ExperimentConfig with defaults for the metric
    """
    config = ExperimentConfig(metric_name=metric_name)
    config.model.model_type = model_type

    # Metric-specific adjustments
    if metric_name == "cpu":
        config.model.hidden_size = 128
        config.model.num_layers = 2
        config.model.dropout = 0.3
        config.model.bidirectional = True
    elif metric_name == "memory":
        config.model.hidden_size = 64
        config.model.num_layers = 2
        config.model.dropout = 0.2
        config.model.bidirectional = False
    elif "disk" in metric_name:
        config.model.hidden_size = 64
        config.model.num_layers = 2
        config.model.dropout = 0.25
    elif "network" in metric_name:
        config.model.hidden_size = 96
        config.model.num_layers = 2
        config.model.dropout = 0.3

    return config


if __name__ == "__main__":
    # Example: Create and save default configs
    metrics = ["cpu", "memory", "disk_reads", "disk_writes", "network_rx", "network_tx"]

    for metric in metrics:
        config = create_default_config(metric, model_type="lstm")
        config_path = f"/home/thoth/dpn/predictive-autoscaling/src/config/model_configs/{metric}_config.yaml"
        config.save_yaml(config_path)
        print(f"Created config for {metric}")
