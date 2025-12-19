"""
MLflow Model Logging and Registry Utilities

Handles logging of models (LSTM, ARIMA, Prophet) to MLflow with proper
signatures, artifacts, and Model Registry integration.
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Any, List
import tempfile
import pickle


class MLflowModelLogger:
    """
    Unified MLflow model logging for all model types.

    Features:
    - Automatic model signature inference
    - Model Registry integration
    - Checkpoint and artifact logging
    - Metadata and tags management
    """

    def __init__(self, mlflow_module, metric_name: str, model_type: str):
        """
        Initialize MLflow model logger.

        Args:
            mlflow_module: The mlflow module (imported)
            metric_name: Name of the metric (cpu, memory, etc.)
            model_type: Type of model (lstm, arima, prophet)
        """
        self.mlflow = mlflow_module
        self.metric_name = metric_name
        self.model_type = model_type

    def create_model_signature(
        self,
        input_example: np.ndarray,
        output_example: Dict[int, np.ndarray]
    ):
        """
        Create MLflow model signature from examples.

        Args:
            input_example: Sample input array (e.g., X_val[0:1])
            output_example: Sample output dict {horizon: predictions}

        Returns:
            MLflow ModelSignature
        """
        from mlflow.models.signature import infer_signature

        # For multi-horizon models, we'll use the first horizon as representative output
        # In practice, the model returns a dict, but for signature we show one horizon
        first_horizon = sorted(output_example.keys())[0]
        representative_output = output_example[first_horizon]

        signature = infer_signature(input_example, representative_output)
        return signature

    def log_pytorch_model(
        self,
        model: torch.nn.Module,
        artifact_path: str,
        config: Any,
        X_scaler: Any,
        y_scaler: Dict[int, Any],
        input_example: Optional[np.ndarray] = None,
        output_example: Optional[Dict[int, np.ndarray]] = None,
        registered_model_name: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> str:
        """
        Log PyTorch model (LSTM) to MLflow.

        Args:
            model: PyTorch model
            artifact_path: Path within MLflow run to store model
            config: Experiment configuration
            X_scaler: Input feature scaler
            y_scaler: Dictionary of output scalers per horizon
            input_example: Optional input example for signature
            output_example: Optional output example for signature
            registered_model_name: Optional name for Model Registry
            checkpoint_path: Optional path to checkpoint file to log

        Returns:
            Model URI
        """
        # Create artifacts dict to save alongside model
        artifacts = {
            "config": config,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
        }

        # Save artifacts to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_path = os.path.join(tmpdir, "model_artifacts.pkl")
            with open(artifacts_path, "wb") as f:
                pickle.dump(artifacts, f)

            # Log artifacts
            self.mlflow.log_artifact(artifacts_path, artifact_path="artifacts")

        # Create signature if examples provided
        signature = None
        if input_example is not None and output_example is not None:
            signature = self.create_model_signature(input_example, output_example)

        # Log the model
        # Note: input_example should be numpy array, not tensor
        model_info = self.mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,  # Keep as numpy array
            registered_model_name=registered_model_name,
            pip_requirements=[
                "torch>=2.0.0",
                "numpy>=1.24.0",
            ],
        )

        # Log checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        return model_info.model_uri

    def log_statistical_model(
        self,
        model: Any,
        artifact_path: str,
        config: Any,
        X_scaler: Any,
        y_scaler: Dict[int, Any],
        model_path: str,
        input_example: Optional[np.ndarray] = None,
        output_example: Optional[Dict[int, np.ndarray]] = None,
        registered_model_name: Optional[str] = None,
    ) -> str:
        """
        Log statistical model (ARIMA/Prophet) to MLflow.

        Args:
            model: Statistical model instance
            artifact_path: Path within MLflow run to store model
            config: Experiment configuration
            X_scaler: Input feature scaler
            y_scaler: Dictionary of output scalers per horizon
            model_path: Path to saved model file (.pkl)
            input_example: Optional input example for signature
            output_example: Optional output example for signature
            registered_model_name: Optional name for Model Registry

        Returns:
            Model URI
        """
        # Create artifacts dict
        artifacts = {
            "config": config,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
            "model_type": self.model_type,
            "metric_name": self.metric_name,
        }

        # Save artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_path = os.path.join(tmpdir, "model_artifacts.pkl")
            with open(artifacts_path, "wb") as f:
                pickle.dump(artifacts, f)

            # Log artifacts
            self.mlflow.log_artifact(artifacts_path, artifact_path="artifacts")

        # Log the model file itself
        self.mlflow.log_artifact(model_path, artifact_path=artifact_path)

        # Create signature if examples provided
        signature = None
        if input_example is not None and output_example is not None:
            signature = self.create_model_signature(input_example, output_example)

        # Create wrapper class dynamically that inherits from PythonModel
        wrapper_instance = self._create_statistical_wrapper(
            model=model,
            model_type=self.model_type,
            scalers={"X_scaler": X_scaler, "y_scaler": y_scaler},
        )

        # For statistical models, we use pyfunc to create a custom wrapper
        # This allows us to log any model type in a consistent way
        model_info = self.mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=wrapper_instance,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
            pip_requirements=[
                "statsmodels>=0.14.0" if self.model_type == "arima" else "prophet>=1.1.0",
                "numpy>=1.24.0",
                "pandas>=2.0.0",
            ],
        )

        return model_info.model_uri

    def _create_statistical_wrapper(
        self, model: Any, model_type: str, scalers: Dict[str, Any]
    ):
        """
        Create a PythonModel wrapper instance for statistical models.

        Args:
            model: The statistical model
            model_type: Type of model (arima/prophet)
            scalers: Dictionary with X_scaler and y_scaler

        Returns:
            Instance of a class that inherits from mlflow.pyfunc.PythonModel
        """
        # Dynamically create a class that inherits from PythonModel
        class StatisticalModelWrapper(self.mlflow.pyfunc.PythonModel):
            """MLflow pyfunc wrapper for statistical models."""

            def __init__(self, model_instance, model_type_str, scalers_dict):
                self.model = model_instance
                self.model_type = model_type_str
                self.scalers = scalers_dict

            def predict(self, context, model_input):
                """Generate predictions."""
                # Convert to numpy if needed
                if hasattr(model_input, "values"):
                    model_input = model_input.values

                # For statistical models, predict all horizons
                horizons = sorted(self.scalers["y_scaler"].keys())
                predictions = self.model.predict_multi_horizon(horizons)

                # Return first horizon's predictions as representative
                return predictions[horizons[0]]

        return StatisticalModelWrapper(model, model_type, scalers)

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> Any:
        """
        Register model in MLflow Model Registry.

        Args:
            model_uri: URI of the logged model
            model_name: Name to register under
            tags: Optional tags for the model version
            description: Optional description

        Returns:
            Registered model version
        """
        # Register the model
        model_version = self.mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        # Create client to update tags and description
        client = self.mlflow.tracking.MlflowClient()

        # Set tags
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value,
                )

        # Set description
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description,
            )

        return model_version

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ):
        """
        Transition model to a different stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (None, Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage
        """
        client = self.mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing,
        )


def create_model_name(metric_name: str, model_type: str, container_name: str = "webapp") -> str:
    """
    Create standardized model name for MLflow Registry.

    Args:
        metric_name: Metric name (cpu, memory, etc.)
        model_type: Model type (lstm, arima, prophet)
        container_name: Container name

    Returns:
        Model name string
    """
    return f"{container_name}_{metric_name}_{model_type}"


def get_model_tags(config: Any, best_val_loss: Optional[float] = None) -> Dict[str, str]:
    """
    Generate standard tags for model version.

    Args:
        config: Experiment configuration
        best_val_loss: Best validation loss (if available)

    Returns:
        Dictionary of tags
    """
    tags = {
        "metric": config.metric_name,
        "model_type": config.model.model_type,
        "container": config.container_name,
        "normalization": config.data.normalization,
        "window_size": str(config.data.window_size),
    }

    if best_val_loss is not None:
        tags["best_val_loss"] = f"{best_val_loss:.6f}"

    # Add model-specific tags
    if config.model.model_type == "lstm":
        tags["hidden_size"] = str(config.model.hidden_size)
        tags["num_layers"] = str(config.model.num_layers)
        tags["dropout"] = str(config.model.dropout)

    return tags


def get_model_description(config: Any, results: Optional[Dict] = None) -> str:
    """
    Generate model description for registry.

    Args:
        config: Experiment configuration
        results: Optional evaluation results

    Returns:
        Description string
    """
    desc = f"Time series forecasting model for {config.metric_name} metric.\n"
    desc += f"Model type: {config.model.model_type.upper()}\n"
    desc += f"Container: {config.container_name}\n"
    desc += f"Prediction horizons: {config.data.prediction_horizons}\n"

    if results and "overall" in results:
        desc += "\nEvaluation Metrics:\n"
        for metric, value in results["overall"].items():
            desc += f"  {metric}: {value:.4f}\n"

    return desc
