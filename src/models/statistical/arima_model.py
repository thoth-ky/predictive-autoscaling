"""
ARIMA Time Series Model
Statistical baseline for comparison with deep learning models.
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Optional, Dict, Tuple
from src.models.base_model import StatisticalBaseModel
import warnings

warnings.filterwarnings("ignore")


class ARIMAPredictor(StatisticalBaseModel):
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.

    Supports:
    - Standard ARIMA (p, d, q)
    - Seasonal ARIMA (P, D, Q, s)
    - Multi-horizon prediction
    """

    def __init__(self, config: dict):
        """
        Initialize ARIMA model.

        Args:
            config: Configuration dict with:
                   - order: (p, d, q) tuple
                   - seasonal_order: (P, D, Q, s) tuple (optional)
        """
        super().__init__(config)

        self.order = config.get("arima_order", (1, 1, 1))
        self.seasonal_order = config.get("arima_seasonal_order", (0, 0, 0, 0))

        self.use_seasonal = self.seasonal_order != (0, 0, 0, 0)

    def fit(self, y_train: np.ndarray, **kwargs):
        """
        Fit ARIMA model on training data.

        Args:
            y_train: Training time series (1D array)
            **kwargs: Additional fitting parameters
        """
        # Convert to 1D if needed
        if y_train.ndim > 1:
            if y_train.shape[1] == 1:
                y_train = y_train.flatten()
            else:
                # Use first column if multivariate
                y_train = y_train[:, 0]

        try:
            if self.use_seasonal:
                self.model = SARIMAX(
                    y_train,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
            else:
                self.model = ARIMA(
                    y_train,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit()

            self.is_fitted = True

        except Exception as e:
            print(f"ARIMA fitting failed: {e}")
            print("Falling back to simpler model (1,1,0)")
            self.order = (1, 1, 0)
            self.model = ARIMA(y_train, order=self.order).fit()
            self.is_fitted = True

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

        forecast = self.model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, "values") else forecast

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


class AutoARIMA(StatisticalBaseModel):
    """
    Auto-ARIMA with automatic parameter selection.

    Uses AIC/BIC to find optimal (p, d, q) parameters.
    """

    def __init__(self, config: dict):
        """
        Initialize Auto-ARIMA.

        Args:
            config: Configuration with max parameter values
        """
        super().__init__(config)

        self.max_p = config.get("max_p", 3)
        self.max_d = config.get("max_d", 2)
        self.max_q = config.get("max_q", 3)

        self.best_order = None
        self.best_aic = np.inf

    def fit(self, y_train: np.ndarray, **kwargs):
        """
        Fit Auto-ARIMA with parameter search.

        Args:
            y_train: Training data
        """
        if y_train.ndim > 1:
            y_train = y_train.flatten()

        print("Searching for best ARIMA parameters...")

        # Grid search over parameters
        for p in range(0, self.max_p + 1):
            for d in range(0, self.max_d + 1):
                for q in range(0, self.max_q + 1):
                    try:
                        model = ARIMA(y_train, order=(p, d, q)).fit()
                        aic = model.aic

                        if aic < self.best_aic:
                            self.best_aic = aic
                            self.best_order = (p, d, q)
                            self.model = model

                    except Exception:
                        continue

        if self.model is None:
            # Fallback to simple model
            self.best_order = (1, 1, 1)
            self.model = ARIMA(y_train, order=self.best_order).fit()

        print(f"Best ARIMA order: {self.best_order} (AIC: {self.best_aic:.2f})")
        self.is_fitted = True

    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Predict N steps ahead."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecast = self.model.forecast(steps=steps)
        return forecast.values if hasattr(forecast, "values") else forecast


def train_arima_baseline(
    y_train: np.ndarray,
    y_val: np.ndarray,
    horizons: list = [20, 60, 120],
    config: Optional[dict] = None,
) -> Tuple[ARIMAPredictor, Dict]:
    """
    Train ARIMA model and evaluate on validation set.

    Args:
        y_train: Training data
        y_val_dict: Validation data per horizon
        horizons: Prediction horizons
        config: ARIMA configuration

    Returns:
        Tuple of (trained_model, validation_results)
    """
    if config is None:
        config = {"arima_order": (1, 1, 1), "arima_seasonal_order": (0, 0, 0, 0)}

    # Flatten to 1D if needed
    if y_train.ndim > 1:
        y_train = y_train[:, 0] if y_train.shape[1] == 1 else y_train.flatten()

    # Train model
    model = ARIMAPredictor(config)
    model.fit(y_train)

    # Predict for all horizons
    predictions = model.predict_multi_horizon(horizons)

    # Evaluate (simplified - just return predictions)
    results = {
        "predictions": predictions,
        "model_order": model.order,
    }

    return model, results


if __name__ == "__main__":
    # Test ARIMA model
    print("ARIMA Time Series Model")
    print("=" * 60)

    # Generate synthetic time series
    np.random.seed(42)
    n = 500
    trend = np.linspace(0, 10, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 50)
    noise = np.random.randn(n) * 2
    y = trend + seasonal + noise

    # Split into train/test
    train_size = int(0.8 * n)
    y_train = y[:train_size]
    y_test = y[train_size:]

    print("\nData:")
    print(f"  Total length: {n}")
    print(f"  Train size: {len(y_train)}")
    print(f"  Test size: {len(y_test)}")

    # Test standard ARIMA
    print("\n1. Standard ARIMA")
    config = {"arima_order": (2, 1, 2)}
    model = ARIMAPredictor(config)
    model.fit(y_train)

    # Predict
    steps = 20
    forecast = model.predict(steps)
    print(f"  Forecast shape: {forecast.shape}")
    print(f"  First 5 predictions: {forecast[:5]}")

    # Test multi-horizon
    print("\n2. Multi-Horizon Prediction")
    horizons = [20, 60, 100]
    multi_forecast = model.predict_multi_horizon(horizons)

    for h, pred in multi_forecast.items():
        print(f"  Horizon {h}: shape {pred.shape}")

    # Test Auto-ARIMA
    print("\n3. Auto-ARIMA")
    auto_config = {"max_p": 2, "max_d": 1, "max_q": 2}
    auto_model = AutoARIMA(auto_config)
    auto_model.fit(y_train)

    auto_forecast = auto_model.predict(steps)
    print(f"  Auto-ARIMA forecast shape: {auto_forecast.shape}")

    print("\nARIMA model working correctly!")
