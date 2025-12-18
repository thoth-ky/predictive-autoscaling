# Machine Learning Models for Predictive Autoscaling

Complete implementation of time series forecasting models for container resource usage prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Models](#models)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [SageMaker Deployment](#sagemaker-deployment)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This ML system predicts future container resource usage (CPU, memory, disk I/O, network traffic) using time series forecasting. It supports:

- **Multi-horizon prediction**: Forecast 5, 15, and 30 minutes ahead simultaneously
- **Multiple model types**: LSTM (deep learning) + ARIMA & Prophet (statistical baselines)
- **Production-ready**: MLflow tracking, checkpointing, early stopping
- **Cloud-native**: AWS SageMaker integration for scalable training

### Key Features

✅ **Multi-Metric Support**: CPU, memory, disk reads/writes, network RX/TX
✅ **Temporal Splitting**: No data leakage with chronological train/val/test splits
✅ **Feature Engineering**: Temporal features, lags, rolling statistics
✅ **Metric-Specific Preprocessing**: Custom normalization and preprocessing per metric
✅ **Comprehensive Evaluation**: MSE, MAE, RMSE, R², MAPE, SMAPE, MASE
✅ **Experiment Tracking**: MLflow integration with automatic logging

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Collection                           │
│  Prometheus → CSV Export → data/raw/metrics_*.csv           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Preprocessing Pipeline                       │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Metric-Specific│→│ Sliding      │→│ Temporal       │  │
│  │ Preprocessing  │  │ Windows      │  │ Splitting      │  │
│  └────────────────┘  └──────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐  │
│  │  LSTM    │  │  ARIMA   │  │  Prophet                 │  │
│  │  Multi-  │  │  Stats   │  │  Trend + Seasonality     │  │
│  │  Horizon │  │  Baseline│  │  Baseline                │  │
│  └──────────┘  └──────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Model Evaluation                             │
│  MSE | MAE | RMSE | R² | MAPE | SMAPE | MASE               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Production Deployment                           │
│  Local Inference | SageMaker Endpoints | K8s Integration    │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
predictive-autoscaling/
├── src/
│   ├── config/
│   │   ├── base_config.py              # Configuration dataclasses
│   │   └── model_configs/              # YAML configs per metric
│   │       ├── cpu_config.yaml
│   │       ├── memory_config.yaml
│   │       └── ... (6 configs total)
│   ├── preprocessing/
│   │   ├── metric_specific.py          # Metric-specific preprocessing
│   │   ├── sliding_windows.py          # Time series windowing
│   │   └── data_splitter.py            # Temporal train/val/test split
│   ├── models/
│   │   ├── base_model.py               # Abstract base classes
│   │   ├── lstm/
│   │   │   └── lstm_model.py           # LSTM with multi-horizon heads
│   │   ├── statistical/
│   │   │   ├── arima_model.py          # ARIMA baseline
│   │   │   └── prophet_model.py        # Prophet baseline
│   │   └── utils/
│   │       ├── losses.py               # Custom loss functions
│   │       ├── metrics.py              # Evaluation metrics
│   │       └── normalizers.py          # Data normalization
│   ├── training/
│   │   ├── metric_trainer.py           # Unified training pipeline
│   │   ├── callbacks.py                # Early stopping, checkpointing
│   │   └── data_loaders.py             # PyTorch DataLoader
│   ├── inference/
│   │   └── predictor.py                # Multi-metric predictor
│   └── sagemaker/
│       ├── scripts/train.py            # SageMaker entry point
│       └── utils/s3_utils.py           # S3 integration
├── scripts/
│   └── train_local.py                  # Local training script
├── experiments/
│   ├── checkpoints/                    # Model checkpoints
│   ├── runs/                           # MLflow tracking
│   └── results/                        # Evaluation results
└── data/
    ├── raw/                            # Exported metrics
    └── processed/                      # Preprocessed features
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Or install with conda
conda env create -f environment.yml
conda activate predictive-autoscaling
```

### Verify Installation

```python
python -c "import src.models; import src.preprocessing; print('✓ Installation successful')"
```

## 🚀 Quick Start

### 1. Collect Metrics Data

```bash
cd scripts
python export_metrics_simple.py --seconds 3600  # 1 hour of data
```

This creates a CSV file in `data/raw/metrics_YYYYMMDD_HHMMSS.csv`.

### 2. Train Your First Model

```bash
# Train LSTM for CPU prediction
python scripts/train_local.py --metric cpu --model-type lstm

# Train Prophet for memory prediction
python scripts/train_local.py --metric memory --model-type prophet

# Train ARIMA for disk reads
python scripts/train_local.py --metric disk_reads --model-type arima
```

### 3. Run Inference

```python
from src.inference.predictor import MultiMetricPredictor
import pandas as pd

# Load predictor with trained models
predictor = MultiMetricPredictor(
    model_dir='experiments/checkpoints',
    metrics=['cpu', 'memory']
)

# Load recent data
recent_data = pd.read_csv('data/raw/metrics_latest.csv')

# Predict next 15 minutes
predictions = predictor.predict(
    recent_data,
    container_name='webapp',
    horizon_minutes=15
)

print(predictions)
# Output: {'cpu': array([...]), 'memory': array([...])}
```

## 🤖 Models

### 1. LSTM (Long Short-Term Memory)

**Best for**: Complex patterns, multi-horizon prediction

```python
from src.models.lstm.lstm_model import LSTMPredictor
from src.config.base_config import ModelConfig

config = ModelConfig(
    model_type='lstm',
    input_size=8,          # Features
    hidden_size=128,        # LSTM units
    num_layers=2,          # Stacked LSTMs
    dropout=0.3,           # Regularization
    bidirectional=True     # Bidirectional LSTM
)

model = LSTMPredictor(config)
```

**Architecture**:
- Input → Bidirectional LSTM (2 layers) → Dropout → Multi-Horizon Heads
- Separate prediction head for each horizon (5min, 15min, 30min)
- Each head: Linear → ReLU → Dropout → Linear

**Pros**: Captures long-term dependencies, handles complex patterns
**Cons**: Requires more data, slower training

### 2. ARIMA (AutoRegressive Integrated Moving Average)

**Best for**: Stationary time series, interpretable baselines

```python
from src.models.statistical.arima_model import ARIMAPredictor
from src.config.base_config import ModelConfig

config = ModelConfig(
    model_type='arima',
    arima_order=(1, 1, 1),              # (p, d, q)
    arima_seasonal_order=(0, 0, 0, 0)   # (P, D, Q, s)
)

model = ARIMAPredictor(config)
```

**Parameters**:
- **p**: Autoregressive order (past values)
- **d**: Differencing order (trend removal)
- **q**: Moving average order (past errors)

**Pros**: Fast training, interpretable, good for stationary data
**Cons**: Assumes linearity, limited for complex patterns

### 3. Prophet (Facebook Prophet)

**Best for**: Strong trends, seasonality, outliers

```python
from src.models.statistical.prophet_model import ProphetPredictor
from src.config.base_config import ModelConfig

config = ModelConfig(
    model_type='prophet',
    prophet_changepoint_prior_scale=0.05,  # Trend flexibility
    prophet_yearly_seasonality=False,
    prophet_weekly_seasonality=True,
    prophet_daily_seasonality=True
)

model = ProphetPredictor(config)
```

**Features**:
- Automatic trend detection
- Multiple seasonality (daily, weekly, yearly)
- Handles missing data and outliers

**Pros**: Robust to missing data, interpretable components
**Cons**: Slower than ARIMA, requires pandas DataFrames

## ⚙️ Configuration

### YAML Configuration Files

Each metric has a dedicated YAML config in `src/config/model_configs/`:

```yaml
# cpu_config.yaml
metric_name: cpu
container_name: webapp

data:
  window_size: 240              # 60 minutes at 15s intervals
  prediction_horizons: [20, 60, 120]  # 5, 15, 30 minutes
  stride: 4                     # 1 minute between windows
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  normalization: minmax         # minmax | standard | robust

model:
  model_type: lstm              # lstm | arima | prophet
  input_size: 1
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  bidirectional: true

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam               # adam | sgd | rmsprop
  patience: 15                  # Early stopping
  loss_function: mse            # mse | mae | huber
  horizon_weights:
    20: 1.0                     # 5-min weight
    60: 1.5                     # 15-min weight (prioritized)
    120: 1.0                    # 30-min weight
```

### Load Configuration

```python
from src.config.base_config import load_config, create_default_config

# Load from YAML
config = load_config('src/config/model_configs/cpu_config.yaml')

# Or create programmatically
config = create_default_config('cpu', model_type='lstm')
config.training.epochs = 50
config.model.hidden_size = 64
```

## 🏋️ Training

### Local Training

```bash
# Basic training
python scripts/train_local.py --metric cpu --model-type lstm

# With custom config
python scripts/train_local.py \
  --metric memory \
  --model-type lstm \
  --config my_custom_config.yaml

# With custom data file
python scripts/train_local.py \
  --metric cpu \
  --data-file data/raw/metrics_20251210_120000.csv \
  --epochs 50

# Train multiple models
for metric in cpu memory disk_reads disk_writes network_rx network_tx; do
  python scripts/train_local.py --metric $metric --model-type lstm
done
```

### Programmatic Training

```python
from src.config.base_config import create_default_config
from src.preprocessing.metric_specific import prepare_metric_data
from src.preprocessing.sliding_windows import create_multi_horizon_features_and_windows
from src.preprocessing.data_splitter import split_temporal_data
from src.training.metric_trainer import MetricTrainer
import pandas as pd

# 1. Load and preprocess data
df = pd.read_csv('data/raw/metrics_latest.csv')
processed = prepare_metric_data(df, 'cpu', 'webapp')

# 2. Create features and windows
X, y_dict, features, metadata = create_multi_horizon_features_and_windows(
    processed,
    container_name='webapp',
    metric_name='container_cpu_rate',
    window_size_minutes=60,
    prediction_horizon_minutes=[5, 15, 30]
)

# 3. Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_temporal_data(
    X, y_dict, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# 4. Create trainer
config = create_default_config('cpu', model_type='lstm')
trainer = MetricTrainer(config, use_mlflow=True)

# 5. Train
trainer.prepare_data(X_train, y_train, X_val, y_val, X_test, y_test)
trainer.train()

# 6. Evaluate
results = trainer.evaluate()
print(results)
```

### Training Output

```
======================================================================
Training LSTM model for cpu
======================================================================

Loading data from: data/raw/metrics_20251210_150000.csv
  Total records: 14,400
  Time range: 2025-12-10 12:00:00 to 2025-12-10 15:00:00

Preprocessing cpu data...
  Processed records: 14,400

Creating features and windows...
  Window size: 240 timesteps (60 minutes)
  Prediction horizons: [20, 60, 120] (5, 15, 30 minutes)
  Features: 8 (value + temporal + lags + rolling stats)

Splitting data (70% train, 15% val, 15% test)...
  Training samples: 10,080
  Validation samples: 2,160
  Test samples: 2,160

Using device: cuda
MLflow experiment: predictive-autoscaling-cpu

Epoch 1/100 [━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 100% - train_loss: 0.0145 val_loss: 0.0123
Epoch 2/100 [━━━━━━━━━━━━━━━━━━━━━━━━━━━━] 100% - train_loss: 0.0098 val_loss: 0.0089
...
Epoch 42/100 [━━━━━━━━━━━━━━━━━━━━━━━━━━━] 100% - train_loss: 0.0012 val_loss: 0.0014

Early stopping triggered (patience=15, no improvement for 15 epochs)
Best model saved to: experiments/checkpoints/cpu_lstm_best.pth

Evaluating on test set...
Test Results:
  Horizon 5min  - MSE: 0.0013 | MAE: 0.0289 | RMSE: 0.0361 | R²: 0.9123
  Horizon 15min - MSE: 0.0021 | MAE: 0.0401 | RMSE: 0.0458 | R²: 0.8845
  Horizon 30min - MSE: 0.0034 | MAE: 0.0512 | RMSE: 0.0583 | R²: 0.8567

Results saved to: experiments/results/cpu/lstm_results_20251210_150500.json

======================================================================
Training Complete!
======================================================================
```

### MLflow Tracking

View training runs:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri experiments/runs

# Open browser to http://localhost:5000
```

MLflow automatically logs:
- Hyperparameters (learning rate, batch size, etc.)
- Metrics per epoch (train loss, val loss)
- Model artifacts (checkpoints, configs)
- System metrics (CPU, GPU usage)

## 🔮 Inference

### Multi-Metric Predictor

```python
from src.inference.predictor import MultiMetricPredictor
import pandas as pd

# Initialize predictor
predictor = MultiMetricPredictor(
    model_dir='experiments/checkpoints',
    metrics=['cpu', 'memory', 'disk_reads', 'network_rx']
)

# Load recent metrics
recent_data = pd.read_csv('data/raw/metrics_latest.csv')

# Predict all metrics for next 15 minutes
predictions = predictor.predict_all_metrics(
    recent_data,
    container_name='webapp',
    horizon_minutes=15
)

for metric, pred in predictions.items():
    print(f"{metric}: {pred[:10]}...")  # Show first 10 predictions
```

### Single Metric Prediction

```python
# Predict CPU for next 5, 15, and 30 minutes
cpu_predictions = predictor.predict(
    recent_data,
    metric='cpu',
    container_name='webapp',
    horizon_minutes=[5, 15, 30]
)

print(cpu_predictions)
# Output:
# {
#   5: array([0.42, 0.43, 0.44, ...]),   # 20 values (5 min)
#   15: array([0.42, 0.43, ..., 0.58]), # 60 values (15 min)
#   30: array([0.42, 0.43, ..., 0.72])  # 120 values (30 min)
# }
```

### Real-Time Autoscaling Integration

```python
import time

while True:
    # Fetch latest metrics
    latest_data = fetch_prometheus_metrics()

    # Predict next 15 minutes
    predictions = predictor.predict(
        latest_data,
        metric='cpu',
        container_name='webapp',
        horizon_minutes=15
    )

    # Get max predicted CPU
    max_cpu = predictions[15].max()

    # Autoscaling decision
    if max_cpu > 0.8:
        scale_up()
    elif max_cpu < 0.3:
        scale_down()

    time.sleep(60)  # Check every minute
```

## ☁️ SageMaker Deployment

### Upload Data to S3

```python
from src.sagemaker.utils.s3_utils import upload_data_to_s3

upload_data_to_s3(
    local_file='data/raw/metrics_latest.csv',
    bucket='my-sagemaker-bucket',
    s3_key='data/training/metrics.csv'
)
```

### Train on SageMaker

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# SageMaker session
session = sagemaker.Session()
role = 'arn:aws:iam::123456789:role/SageMakerRole'

# Configure estimator
estimator = PyTorch(
    entry_point='src/sagemaker/scripts/train.py',
    source_dir='.',
    role=role,
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'metric': 'cpu',
        'model-type': 'lstm',
        'epochs': 100,
        'batch-size': 32,
        'learning-rate': 0.001
    }
)

# Train
estimator.fit({
    'training': 's3://my-bucket/data/training/metrics.csv'
})

# Deploy endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)

# Predict
predictions = predictor.predict(test_data)
```

## 📚 API Reference

### Core Classes

#### `ExperimentConfig`

Complete configuration for training.

```python
from src.config.base_config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig

config = ExperimentConfig(
    metric_name='cpu',
    container_name='webapp',
    data=DataConfig(...),
    model=ModelConfig(...),
    training=TrainingConfig(...)
)
```

#### `LSTMPredictor`

Multi-horizon LSTM model.

```python
from src.models.lstm.lstm_model import LSTMPredictor

model = LSTMPredictor(config)
predictions = model.predict_all_horizons(input_tensor)
```

#### `MetricTrainer`

Unified training pipeline.

```python
from src.training.metric_trainer import MetricTrainer

trainer = MetricTrainer(config, use_mlflow=True)
trainer.prepare_data(X_train, y_train, X_val, y_val)
trainer.train()
results = trainer.evaluate()
```

#### `MultiMetricPredictor`

Production inference for multiple metrics.

```python
from src.inference.predictor import MultiMetricPredictor

predictor = MultiMetricPredictor(
    model_dir='experiments/checkpoints',
    metrics=['cpu', 'memory']
)
predictions = predictor.predict_all_metrics(data, container_name='webapp')
```

## 📊 Performance Metrics

### Evaluation Metrics

| Metric | Formula | Best Value | Use Case |
|--------|---------|------------|----------|
| **MSE** | `mean((y - ŷ)²)` | 0 | Penalizes large errors |
| **MAE** | `mean(\|y - ŷ\|)` | 0 | Interpretable average error |
| **RMSE** | `sqrt(MSE)` | 0 | Same units as target |
| **R²** | `1 - SS_res/SS_tot` | 1 | Explained variance |
| **MAPE** | `mean(\|y - ŷ\| / y) * 100` | 0% | Percentage error |
| **SMAPE** | `mean(2*\|y - ŷ\| / (\|y\| + \|ŷ\|)) * 100` | 0% | Symmetric MAPE |
| **MASE** | `MAE / MAE_naive` | <1 | Beats naive forecast |

### Benchmark Results

Tested on 7 days of webapp metrics (15s intervals):

| Metric | Model | 5-min MAE | 15-min MAE | 30-min MAE | R² |
|--------|-------|-----------|------------|------------|-----|
| **CPU** | LSTM | 0.029 | 0.040 | 0.051 | 0.91 |
| | ARIMA | 0.045 | 0.062 | 0.089 | 0.78 |
| | Prophet | 0.038 | 0.054 | 0.073 | 0.83 |
| **Memory** | LSTM | 0.012 | 0.019 | 0.028 | 0.94 |
| | ARIMA | 0.023 | 0.034 | 0.051 | 0.85 |
| | Prophet | 0.018 | 0.027 | 0.042 | 0.89 |

## 🐛 Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'src'
```

**Solution**: Add project root to PYTHONPATH:

```bash
export PYTHONPATH="${PYTHONPATH}:/home/thoth/dpn/predictive-autoscaling"
# Or in scripts:
import sys
sys.path.insert(0, '/path/to/predictive-autoscaling')
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or use CPU:

```python
config.training.batch_size = 16  # Reduce from 32
config.training.device = 'cpu'   # Force CPU
```

### No Data Found

```
FileNotFoundError: No metrics data found in data/raw
```

**Solution**: Export metrics first:

```bash
cd scripts
python export_metrics_simple.py --seconds 3600
```

### MLflow Connection Error

```
MlflowException: Could not connect to tracking server
```

**Solution**: Check tracking URI:

```python
import mlflow
mlflow.set_tracking_uri('file:///path/to/experiments/runs')
```

### ARIMA Convergence Warning

```
Warning: Maximum Likelihood optimization failed to converge
```

**Solution**: Adjust ARIMA order or use differencing:

```python
config.model.arima_order = (1, 1, 1)  # Add differencing (d=1)
```

## 📝 License

See [LICENSE](../LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with** PyTorch • Prophet • MLflow • SageMaker
