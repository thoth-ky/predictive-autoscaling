"""
Evaluation Metrics for Time Series Models
Comprehensive metrics for measuring prediction performance.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Optional


class ModelEvaluator:
    """
    Evaluate time series model predictions with comprehensive metrics.

    Computes:
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² Score
    - Directional Accuracy
    - Horizon degradation analysis
    """

    def __init__(self, metric_name: str = "unknown"):
        """
        Initialize evaluator.

        Args:
            metric_name: Name of metric being evaluated (for logging)
        """
        self.metric_name = metric_name

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true.flatten(), y_pred.flatten())

    def mape(
        self, y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero

        Returns:
            MAPE as percentage
        """
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # Avoid division by zero
        mask = np.abs(y_true_flat) > epsilon
        if not mask.any():
            return 0.0

        return (
            np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask]))
            * 100
        )

    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score."""
        return r2_score(y_true.flatten(), y_pred.flatten())

    def directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct trend predictions).

        Measures if the model correctly predicts whether the value will go up or down.

        Args:
            y_true: True sequences of shape (n_samples, horizon)
            y_pred: Predicted sequences

        Returns:
            Directional accuracy as percentage (0-100)
        """
        if y_true.ndim == 1:
            # Single timestep predictions
            return 100.0  # Not applicable for single values

        # Calculate differences between consecutive timesteps
        true_diff = np.diff(y_true, axis=1)
        pred_diff = np.diff(y_pred, axis=1)

        # Check if signs match (same direction)
        same_direction = np.sign(true_diff) == np.sign(pred_diff)

        return np.mean(same_direction) * 100

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, horizon: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single prediction horizon.

        Args:
            y_true: True values of shape (n_samples, horizon) or (n_samples,)
            y_pred: Predicted values (same shape)
            horizon: Horizon identifier (for logging)

        Returns:
            Dictionary of metric names to values
        """
        metrics = {
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "mape": self.mape(y_true, y_pred),
            "r2_score": self.r2(y_true, y_pred),
        }

        # Add directional accuracy if multi-step predictions
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            metrics["directional_accuracy"] = self.directional_accuracy(y_true, y_pred)

        if horizon is not None:
            # Add horizon identifier to metric names
            metrics = {f"{k}_h{horizon}": v for k, v in metrics.items()}

        return metrics

    def horizon_analysis(
        self, y_true_dict: Dict[int, np.ndarray], y_pred_dict: Dict[int, np.ndarray]
    ) -> Dict[str, any]:
        """
        Analyze performance across multiple prediction horizons.

        Args:
            y_true_dict: Dict mapping horizon to true values
                        {20: (n_samples, 20), 60: (n_samples, 60), ...}
            y_pred_dict: Dict mapping horizon to predictions

        Returns:
            Dictionary with per-horizon metrics and degradation analysis
        """
        results = {
            "by_horizon": {},
            "degradation": {},
        }

        # Compute metrics for each horizon
        for horizon in sorted(y_true_dict.keys()):
            y_true = y_true_dict[horizon]
            y_pred = y_pred_dict[horizon]

            horizon_minutes = horizon // 4  # Convert timesteps to minutes
            results["by_horizon"][horizon_minutes] = self.compute_metrics(
                y_true, y_pred, horizon=horizon_minutes
            )

        # Analyze degradation (how error increases with horizon)
        horizons_sorted = sorted(y_true_dict.keys())
        rmse_values = []
        mae_values = []

        for h in horizons_sorted:
            y_true = y_true_dict[h]
            y_pred = y_pred_dict[h]
            rmse_values.append(self.rmse(y_true, y_pred))
            mae_values.append(self.mae(y_true, y_pred))

        results["degradation"]["horizons"] = [h // 4 for h in horizons_sorted]
        results["degradation"]["rmse_progression"] = rmse_values
        results["degradation"]["mae_progression"] = mae_values

        # Calculate degradation rate (error increase per minute)
        if len(horizons_sorted) > 1:
            h1, h2 = horizons_sorted[0] // 4, horizons_sorted[-1] // 4
            rmse_increase = (rmse_values[-1] - rmse_values[0]) / (h2 - h1)
            mae_increase = (mae_values[-1] - mae_values[0]) / (h2 - h1)

            results["degradation"]["rmse_per_minute"] = rmse_increase
            results["degradation"]["mae_per_minute"] = mae_increase

        return results

    def per_timestep_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate error at each timestep in the prediction horizon.

        Useful for understanding which parts of the horizon are harder to predict.

        Args:
            y_true: Shape (n_samples, horizon)
            y_pred: Shape (n_samples, horizon)

        Returns:
            Array of MAE per timestep (length = horizon)
        """
        if y_true.ndim == 1:
            # Single timestep
            return np.array([self.mae(y_true, y_pred)])

        horizon = y_true.shape[1]
        errors = []

        for t in range(horizon):
            error_t = np.mean(np.abs(y_true[:, t] - y_pred[:, t]))
            errors.append(error_t)

        return np.array(errors)

    def print_evaluation_report(
        self, y_true_dict: Dict[int, np.ndarray], y_pred_dict: Dict[int, np.ndarray]
    ):
        """
        Print comprehensive evaluation report.

        Args:
            y_true_dict: True values per horizon
            y_pred_dict: Predictions per horizon
        """
        print(f"\n{'=' * 60}")
        print(f"Evaluation Report: {self.metric_name}")
        print(f"{'=' * 60}")

        results = self.horizon_analysis(y_true_dict, y_pred_dict)

        # Print per-horizon metrics
        print("\nPerformance by Horizon:")
        print(f"{'Horizon':<12} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R²':<10}")
        print("-" * 60)

        for horizon_min in sorted(results["by_horizon"].keys()):
            metrics = results["by_horizon"][horizon_min]
            rmse_key = f"rmse_h{horizon_min}"
            mae_key = f"mae_h{horizon_min}"
            mape_key = f"mape_h{horizon_min}"
            r2_key = f"r2_score_h{horizon_min}"

            print(
                f"{horizon_min:>3} min     "
                f"{metrics.get(rmse_key, 0):<10.4f} "
                f"{metrics.get(mae_key, 0):<10.4f} "
                f"{metrics.get(mape_key, 0):<10.2f} "
                f"{metrics.get(r2_key, 0):<10.4f}"
            )

        # Print degradation analysis
        if "rmse_per_minute" in results["degradation"]:
            print("\nError Degradation:")
            print(
                f"  RMSE increase per minute: {results['degradation']['rmse_per_minute']:.4f}"
            )
            print(
                f"  MAE increase per minute: {results['degradation']['mae_per_minute']:.4f}"
            )

        print(f"\n{'=' * 60}\n")


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test_dict: Dict[int, np.ndarray],
    metric_name: str = "unknown",
    device: str = "cpu",
) -> Dict:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model with predict_all_horizons method
        X_test: Test input sequences
        y_test_dict: Test targets per horizon
        metric_name: Name of metric (for logging)
        device: Device to run inference on

    Returns:
        Dictionary with evaluation results
    """
    import torch

    model.eval()
    model.to(device)

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)

        # Get predictions for all horizons
        predictions = model.predict_all_horizons(X_test_tensor)

        # Convert to numpy
        y_pred_dict = {h: pred.cpu().numpy() for h, pred in predictions.items()}

    # Evaluate
    evaluator = ModelEvaluator(metric_name=metric_name)
    results = evaluator.horizon_analysis(y_test_dict, y_pred_dict)

    # Print report
    evaluator.print_evaluation_report(y_test_dict, y_pred_dict)

    return results


if __name__ == "__main__":
    # Test evaluation metrics
    print("Time Series Evaluation Metrics")
    print("=" * 60)

    # Create dummy predictions
    n_samples = 100
    horizons = [20, 60, 120]

    y_true_dict = {h: np.random.randn(n_samples, h) * 10 + 50 for h in horizons}

    # Simulate predictions with some error
    y_pred_dict = {}
    for h in horizons:
        noise = np.random.randn(n_samples, h) * 2  # Add noise
        y_pred_dict[h] = y_true_dict[h] + noise

    # Evaluate
    evaluator = ModelEvaluator(metric_name="cpu")
    evaluator.print_evaluation_report(y_true_dict, y_pred_dict)

    # Test individual metrics
    print("Testing Individual Metrics:")
    y_true = y_true_dict[60]
    y_pred = y_pred_dict[60]

    print(f"  RMSE: {evaluator.rmse(y_true, y_pred):.4f}")
    print(f"  MAE: {evaluator.mae(y_true, y_pred):.4f}")
    print(f"  MAPE: {evaluator.mape(y_true, y_pred):.2f}%")
    print(f"  R²: {evaluator.r2(y_true, y_pred):.4f}")
    print(
        f"  Directional Accuracy: {evaluator.directional_accuracy(y_true, y_pred):.2f}%"
    )

    # Test per-timestep error
    print("\nPer-Timestep Error (first 10 steps):")
    timestep_errors = evaluator.per_timestep_error(y_true, y_pred)
    print(f"  {timestep_errors[:10]}")

    print("\nMetrics module working correctly!")
