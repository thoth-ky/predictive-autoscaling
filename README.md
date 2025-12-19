# Predictive Autoscaling

Time series forecasting system for predicting container resource usage (CPU, memory, disk I/O, network traffic) with multi-horizon predictions and multiple model architectures.

## Features

- **Multi-horizon forecasting**: 5, 15, and 30-minute predictions
- **Multiple models**: LSTM, ARIMA, Prophet
- **Multi-metric support**: CPU, memory, disk I/O, network throughput
- **Production-ready**: MLflow tracking, checkpointing, early stopping
- **Cloud deployment**: AWS SageMaker integration

## Installation

### Requirements

- Python 3.10-3.14
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
pip install -e .
```

## Quick Start

### 1. Export metrics from Prometheus

```bash
python scripts/exporters/export_metrics_targeted.py --seconds 3600
```

### 2. Train a model

```bash
python scripts/train_local.py --metric cpu --model-type lstm
```

### 3. Make predictions

```python
from src.inference.predictor import MultiMetricPredictor
import pandas as pd

predictor = MultiMetricPredictor(
    model_dir='experiments/checkpoints',
    metrics=['cpu', 'memory']
)

recent_data = pd.read_csv('data/raw/metrics_latest.csv')
predictions = predictor.predict(
    recent_data,
    container_name='webapp',
    horizon_minutes=15
)
```

## Models

### LSTM
- **Architecture**: Bidirectional LSTM with multi-horizon prediction heads
- **Best for**: Complex patterns, long-term dependencies
- **Cons**: Requires more data, slower training

### ARIMA
- **Architecture**: AutoRegressive Integrated Moving Average
- **Best for**: Stationary time series, interpretable baselines
- **Cons**: Assumes linearity, limited for complex patterns

### Prophet
- **Architecture**: Facebook's forecasting tool with trend and seasonality components
- **Best for**: Strong trends, multiple seasonality patterns
- **Cons**: Slower than ARIMA

## Configuration

Each metric has a YAML config in [src/config/model_configs/](src/config/model_configs/). Example:

```yaml
metric_name: cpu
container_name: webapp
data:
  window_size: 240
  prediction_horizons: [20, 60, 120]  # 5, 15, 30 minutes
  train_split: 0.7
model:
  model_type: lstm
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## Training

```bash
# Train a model
python scripts/train_local.py --metric cpu --model-type lstm

# Train with custom config
python scripts/train_local.py --metric cpu --config my_config.yaml

# View training runs with MLflow
mlflow ui --backend-store-uri experiments/runs
```

## Usage

### Python API

```python
from src.inference.predictor import MultiMetricPredictor
import pandas as pd

predictor = MultiMetricPredictor(
    model_dir='experiments/checkpoints',
    metrics=['cpu', 'memory']
)

data = pd.read_csv('data/raw/metrics_latest.csv')
predictions = predictor.predict_all_metrics(
    data,
    container_name='webapp',
    horizon_minutes=15
)
```

## AWS SageMaker Deployment

```python
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='src/sagemaker/scripts/train.py',
    source_dir='.',
    role='arn:aws:iam::123456789:role/SageMakerRole',
    instance_type='ml.p3.2xlarge',
    framework_version='2.0',
    py_version='py310',
    hyperparameters={'metric': 'cpu', 'model-type': 'lstm'}
)

estimator.fit({'training': 's3://bucket/data/metrics.csv'})
predictor = estimator.deploy(instance_count=1, instance_type='ml.t2.medium')
```

## Project Structure

```
src/
├── config/          # Configuration management
├── models/          # LSTM, ARIMA, Prophet implementations
├── preprocessing/   # Data preparation and feature engineering
├── training/        # Training pipeline with MLflow tracking
├── inference/       # Multi-metric predictor for production
└── sagemaker/       # AWS SageMaker integration

scripts/             # Training and data export scripts
data/                # Raw and processed metrics
experiments/         # Checkpoints, MLflow runs, results
```

## Performance

Benchmark results (to be updated):

| Metric | Model | 5-min MAE | 15-min MAE | 30-min MAE | R² |
|--------|-------|-----------|------------|------------|-----|
| CPU | LSTM | - | - | - | - |
| CPU | ARIMA | - | - | - | - |
| Memory | LSTM | - | - | - | - |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## License

See [LICENSE](LICENSE) for details.

---

Built with PyTorch, Prophet, MLflow, and AWS SageMaker.
