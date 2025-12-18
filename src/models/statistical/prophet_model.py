"""
Prophet Time Series Model
Facebook Prophet for time series forecasting with trend and seasonality.
"""

import numpy as np
import pandas as pd
from prophet import Prophet
from typing import Optional, Dict, Tuple
from src.models.base_model import StatisticalBaseModel
import warnings
warnings.filterwarnings('ignore')


class ProphetPredictor(StatisticalBaseModel):
    """
    Facebook Prophet model for time series forecasting.

    Advantages:
    - Handles missing data well
    - Automatic seasonality detection
    - Robust to outliers
    - Interpretable components (trend, seasonality, holidays)
    """

    def __init__(self, config: dict):
        """
        Initialize Prophet model.

        Args:
            config: Configuration dict with:
                   - changepoint_prior_scale: Flexibility of trend changes
                   - seasonality settings
        """
        super().__init__(config)

        self.changepoint_prior_scale = config.get('prophet_changepoint_prior_scale', 0.05)
        self.yearly_seasonality = config.get('prophet_yearly_seasonality', False)
        self.weekly_seasonality = config.get('prophet_weekly_seasonality', True)
        self.daily_seasonality = config.get('prophet_daily_seasonality', True)

        self.interval_width = config.get('interval_width', 0.95)
        self.freq = config.get('freq', '15S')  # 15 second intervals

    def fit(self, y_train: np.ndarray, timestamps: Optional[pd.DatetimeIndex] = None, **kwargs):
        """
        Fit Prophet model.

        Args:
            y_train: Training time series (1D or 2D)
            timestamps: Timestamps for the data (required for Prophet)
            **kwargs: Additional parameters
        """
        # Convert to 1D if needed
        if y_train.ndim > 1:
            if y_train.shape[1] == 1:
                y_train = y_train.flatten()
            else:
                y_train = y_train[:, 0]

        # Create timestamps if not provided
        if timestamps is None:
            timestamps = pd.date_range(
                start='2024-01-01',
                periods=len(y_train),
                freq=self.freq
            )

        # Prepare DataFrame for Prophet
        df = pd.DataFrame({
            'ds': timestamps,
            'y': y_train
        })

        # Initialize and fit Prophet model
        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            interval_width=self.interval_width
        )

        # Suppress Prophet logging
        import logging
        logging.getLogger('prophet').setLevel(logging.ERROR)

        self.model.fit(df)
        self.is_fitted = True

        # Store last timestamp for prediction
        self.last_timestamp = timestamps[-1]

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Predict N steps ahead.

        Args:
            steps: Number of timesteps to forecast

        Returns:
            Predictions array of shape (steps,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Create future dataframe
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_timestamp + pd.Timedelta(self.freq),
                periods=steps,
                freq=self.freq
            )
        })

        # Make prediction
        forecast = self.model.predict(future)

        # Return yhat (point predictions)
        return forecast['yhat'].values

    def predict_with_uncertainty(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty intervals.

        Args:
            steps: Number of steps to forecast

        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_timestamp + pd.Timedelta(self.freq),
                periods=steps,
                freq=self.freq
            )
        })

        forecast = self.model.predict(future)

        return (
            forecast['yhat'].values,
            forecast['yhat_lower'].values,
            forecast['yhat_upper'].values
        )

    def predict_multi_horizon(self, horizons: list) -> Dict[int, np.ndarray]:
        """
        Predict for multiple horizons.

        Args:
            horizons: List of horizons (e.g., [20, 60, 120])

        Returns:
            Dict mapping horizon to predictions
        """
        max_horizon = max(horizons)

        # Predict up to max horizon
        full_forecast = self.predict(max_horizon)

        # Extract predictions for each horizon
        predictions = {}
        for h in horizons:
            predictions[h] = full_forecast[:h]

        return predictions

    def plot_forecast(self, steps: int = 120):
        """
        Plot forecast with components (requires matplotlib).

        Args:
            steps: Number of steps to forecast
        """
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=self.last_timestamp + pd.Timedelta(self.freq),
                periods=steps,
                freq=self.freq
            )
        })

        forecast = self.model.predict(future)

        # Plot
        import matplotlib.pyplot as plt

        fig1 = self.model.plot(forecast)
        plt.title('Prophet Forecast')
        plt.tight_layout()

        fig2 = self.model.plot_components(forecast)
        plt.tight_layout()

        plt.show()


def train_prophet_baseline(y_train: np.ndarray,
                           timestamps: Optional[pd.DatetimeIndex] = None,
                           horizons: list = [20, 60, 120],
                           config: Optional[dict] = None) -> Tuple[ProphetPredictor, Dict]:
    """
    Train Prophet model and evaluate.

    Args:
        y_train: Training data
        timestamps: Timestamps for data
        horizons: Prediction horizons
        config: Prophet configuration

    Returns:
        Tuple of (trained_model, results)
    """
    if config is None:
        config = {
            'prophet_changepoint_prior_scale': 0.05,
            'prophet_yearly_seasonality': False,
            'prophet_weekly_seasonality': True,
            'prophet_daily_seasonality': True,
            'freq': '15S'
        }

    # Train model
    model = ProphetPredictor(config)
    model.fit(y_train, timestamps=timestamps)

    # Predict for all horizons
    predictions = model.predict_multi_horizon(horizons)

    results = {
        'predictions': predictions,
        'config': config,
    }

    return model, results


if __name__ == '__main__':
    # Test Prophet model
    print("Prophet Time Series Model")
    print("=" * 60)

    # Generate synthetic time series with trend and seasonality
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range(start='2024-01-01', periods=n, freq='15S')

    # Trend
    trend = np.linspace(50, 70, n)

    # Seasonality (daily pattern)
    hours = timestamps.hour + timestamps.minute / 60
    daily_seasonal = 10 * np.sin(2 * np.pi * hours / 24)

    # Noise
    noise = np.random.randn(n) * 2

    y = trend + daily_seasonal + noise

    # Split into train/test
    train_size = int(0.8 * n)
    y_train = y[:train_size]
    timestamps_train = timestamps[:train_size]

    print(f"\nData:")
    print(f"  Total length: {n}")
    print(f"  Train size: {len(y_train)}")
    print(f"  Time range: {timestamps[0]} to {timestamps[-1]}")

    # Test Prophet
    print("\n1. Training Prophet Model")
    config = {
        'prophet_changepoint_prior_scale': 0.05,
        'prophet_yearly_seasonality': False,
        'prophet_weekly_seasonality': False,
        'prophet_daily_seasonality': True,
        'freq': '15S'
    }

    model = ProphetPredictor(config)
    model.fit(y_train, timestamps=timestamps_train)
    print("  Model fitted successfully!")

    # Test prediction
    print("\n2. Making Predictions")
    steps = 20
    forecast = model.predict(steps)
    print(f"  Forecast shape: {forecast.shape}")
    print(f"  First 5 predictions: {forecast[:5]}")

    # Test with uncertainty
    print("\n3. Prediction with Uncertainty")
    pred, lower, upper = model.predict_with_uncertainty(steps)
    print(f"  Predictions: {pred[:3]}")
    print(f"  Lower bound: {lower[:3]}")
    print(f"  Upper bound: {upper[:3]}")

    # Test multi-horizon
    print("\n4. Multi-Horizon Prediction")
    horizons = [20, 60, 120]
    multi_forecast = model.predict_multi_horizon(horizons)

    for h, pred in multi_forecast.items():
        horizon_minutes = h * 15 / 60  # Convert to minutes
        print(f"  Horizon {horizon_minutes:.1f} min ({h} steps): shape {pred.shape}")

    print("\nProphet model working correctly!")
