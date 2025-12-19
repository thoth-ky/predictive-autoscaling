# Multi-stage Dockerfile for Predictive Autoscaling
# Supports CPU and GPU (CUDA) builds
# Python 3.14.2, CUDA 13.0.2, Ubuntu 24.04

ARG BUILD_TYPE=cpu

# ============================================================================
# Stage 1: Base (CPU)
# ============================================================================
FROM python:3.14.2-slim as base-cpu

# Environment: unbuffered output, no bytecode, no pip cache, non-interactive
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ============================================================================
# Stage 2: Base (GPU)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 as base-gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.14 python3.14-dev python3.14-venv python3-pip \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.14 /usr/bin/python \
    && ln -sf /usr/bin/python3.14 /usr/bin/python3
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

WORKDIR /app

# ============================================================================
# Stage 3: Dependencies
# ============================================================================

FROM base-${BUILD_TYPE} as dependencies

COPY pyproject.toml .

# Install Python dependencies (PyTorch auto-detects CUDA)
RUN pip install --no-cache-dir --ignore-installed .

# ============================================================================
# Stage 4: Application
# ============================================================================
FROM dependencies as application

COPY . .

RUN mkdir -p data/raw data/processed experiments/checkpoints experiments/runs experiments/results logs

ENV PYTHONPATH="${PYTHONPATH}:/app"

# Ports: 5000 (MLflow), 8888 (Jupyter), 6006 (TensorBoard)
EXPOSE 5000 8888 6006

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

LABEL maintainer="Thoth" \
      description="Time series forecasting for predictive container autoscaling" \
      version="1.0"
