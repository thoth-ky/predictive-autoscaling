#!/usr/bin/env python3
"""
SageMaker Training Script
Entry point for training models on AWS SageMaker.

This script follows SageMaker conventions:
- Reads data from /opt/ml/input/data/
- Saves model to /opt/ml/model/
- Logs metrics that SageMaker can capture
"""

import argparse
import os
import sys
import json
import numpy as np
import torch

# SageMaker paths
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
SM_CHANNEL_VAL = os.environ.get(
    "SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"
)
SM_OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# Add src to path
sys.path.insert(0, "/opt/ml/code/src")

from src.models.lstm.lstm_model import LSTMPredictor
from src.models.utils.losses import MultiHorizonLoss
from src.models.utils.normalizers import TimeSeriesNormalizer
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.data_loaders import create_data_loaders
from src.config.base_config import ModelConfig, TrainingConfig, DataConfig


def parse_args():
    """Parse command-line arguments from SageMaker."""
    parser = argparse.ArgumentParser()

    # SageMaker-specific arguments
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    parser.add_argument("--train", type=str, default=SM_CHANNEL_TRAIN)
    parser.add_argument("--validation", type=str, default=SM_CHANNEL_VAL)
    parser.add_argument("--output-data-dir", type=str, default=SM_OUTPUT_DATA_DIR)

    # Hyperparameters
    parser.add_argument("--metric-name", type=str, required=True)
    parser.add_argument("--model-type", type=str, default="lstm")

    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=int, default=0)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=15)

    # Data parameters
    parser.add_argument("--prediction-horizons", type=str, default="20,60,120")

    return parser.parse_args()


def load_data(data_dir: str):
    """
    Load data from SageMaker input directory.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (X, y_dict)
    """
    X_path = os.path.join(data_dir, "X.npy")
    y_dir = os.path.join(data_dir, "y")

    # Load X
    X = np.load(X_path)

    # Load y for each horizon
    y_dict = {}
    for horizon_file in os.listdir(y_dir):
        if horizon_file.endswith(".npy"):
            horizon = int(horizon_file.replace("y_", "").replace(".npy", ""))
            y_dict[horizon] = np.load(os.path.join(y_dir, horizon_file))

    return X, y_dict


def save_model(model, model_dir: str, scalers: dict, config: dict):
    """
    Save model in SageMaker-compatible format.

    Args:
        model: Trained model
        model_dir: Directory to save model
        scalers: Dict with X_scaler and y_scaler
        config: Configuration dict
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save PyTorch model
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    # Save config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Save scalers
    import pickle

    scalers_path = os.path.join(model_dir, "scalers.pkl")
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)

    print(f"Model saved to {model_dir}")


def train(args):
    """
    Main training function.

    Args:
        args: Parsed arguments
    """
    print("Starting SageMaker training...")
    print(f"  Metric: {args.metric_name}")
    print(f"  Model type: {args.model_type}")

    # Parse prediction horizons
    prediction_horizons = [int(h) for h in args.prediction_horizons.split(",")]

    # Load training data
    print("\nLoading training data...")
    X_train, y_train_dict = load_data(args.train)
    X_val, y_val_dict = load_data(args.validation)

    print(f"  X_train shape: {X_train.shape}")
    print(f"  Horizons: {sorted(y_train_dict.keys())}")

    # Normalize data
    print("\nNormalizing data...")
    X_scaler = TimeSeriesNormalizer(method="minmax")
    X_train_norm = X_scaler.fit_transform(X_train)
    X_val_norm = X_scaler.transform(X_val)

    y_scaler = {}
    y_train_norm_dict = {}
    y_val_norm_dict = {}

    for horizon in y_train_dict.keys():
        y_scaler[horizon] = TimeSeriesNormalizer(method="minmax")
        y_train_norm_dict[horizon] = (
            y_scaler[horizon]
            .fit_transform(y_train_dict[horizon].reshape(-1, 1))
            .reshape(y_train_dict[horizon].shape)
        )
        y_val_norm_dict[horizon] = (
            y_scaler[horizon]
            .transform(y_val_dict[horizon].reshape(-1, 1))
            .reshape(y_val_dict[horizon].shape)
        )

    # Create model config
    model_config = ModelConfig(
        model_type="lstm",
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bool(args.bidirectional),
    )
    model_config.prediction_horizons = prediction_horizons

    # Build model
    print("\nBuilding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = LSTMPredictor(model_config)
    model.to(device)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_fn = MultiHorizonLoss(base_loss="mse")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        X_train_norm,
        y_train_norm_dict,
        X_val_norm,
        y_val_norm_dict,
        batch_size=args.batch_size,
    )

    # Setup early stopping
    early_stop = EarlyStopping(patience=args.patience, mode="min")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_X, batch_y_dict in train_loader:
            batch_X = batch_X.to(device)
            batch_y_dict = {h: y.to(device) for h, y in batch_y_dict.items()}

            optimizer.zero_grad()
            predictions = model.predict_all_horizons(batch_X)
            loss = loss_fn(predictions, batch_y_dict)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validate
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y_dict in val_loader:
                batch_X = batch_X.to(device)
                batch_y_dict = {h: y.to(device) for h, y in batch_y_dict.items()}

                predictions = model.predict_all_horizons(batch_X)
                loss = loss_fn(predictions, batch_y_dict)

                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        # Log metrics (SageMaker captures these prints)
        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )
        print(f"Val RMSE: {np.sqrt(val_loss):.6f}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Early stopping
        if early_stop(val_loss, epoch):
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Save model
    print("\nSaving model...")
    save_model(
        model=model,
        model_dir=args.model_dir,
        scalers={"X_scaler": X_scaler, "y_scaler": y_scaler},
        config=model_config.__dict__,
    )

    print("Training complete!")


if __name__ == "__main__":
    args = parse_args()
    train(args)
