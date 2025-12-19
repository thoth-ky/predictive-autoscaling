# Multi-stage Dockerfile for Predictive Autoscaling ML Training
# Supports both CPU and GPU (CUDA) environments

# Use ARG to choose between CPU and GPU base
ARG BUILD_TYPE=cpu

# ============================================================================
# Stage 1: Base Image (CPU version)
# ============================================================================
FROM python:3.14.2-slim as base-cpu

# Set environment variables for Python optimization and build configuration
# Force stdout/stderr to be unbuffered for real-time logging
# Prevent Python from writing .pyc files (saves space & avoids permission issues)
# Disable pip cache to reduce image size
# Skip pip version check to speed up pip operations
# Prevent interactive prompts during apt package installation
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 2: GPU Image (CUDA support)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04 as base-gpu

# Install Python 3.14 from deadsnakes PPA and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.14 \
    python3.14-dev \
    python3.14-venv \
    python3-pip \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.14 /usr/bin/python && \
    ln -sf /usr/bin/python3.14 /usr/bin/python3

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

WORKDIR /app

# ============================================================================
# Stage 3: Dependencies Installation
# ============================================================================

FROM base-${BUILD_TYPE} as dependencies

# Upgrade pip
# RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY pyproject.toml .

# Install Python dependencies
# For GPU builds, PyTorch will automatically use CUDA if available
RUN pip install --no-cache-dir .

# Install additional utilities for development
RUN pip install --no-cache-dir \
    black \
    flake8 \
    pytest \
    pytest-cov

# ============================================================================
# Stage 4: Application
# ============================================================================
FROM dependencies as application

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p \
    data/raw \
    data/processed \
    experiments/checkpoints \
    experiments/runs \
    experiments/results \
    logs

# Set Python path to include src directory
ENV PYTHONPATH="${PYTHONPATH}:/app"

# Expose ports
# 5000 - MLflow UI
# 8888 - Jupyter Notebook
# 6006 - TensorBoard (if needed)
EXPOSE 5000 8888 6006

# Copy and setup entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]

# ============================================================================
# Labels
# ============================================================================
LABEL maintainer="Predictive Autoscaling Team"
LABEL description="ML Training Environment for Predictive Container Autoscaling"
LABEL version="1.0"
