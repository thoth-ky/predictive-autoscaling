#!/bin/bash
set -e

# Entrypoint script for Predictive Autoscaling ML Training
# Works in both container and local environments
# Supports CPU and GPU modes

# Function to display usage
usage() {
    echo "Usage: ./entrypoint.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train              Train models locally"
    echo "  mlflow-ui          Start MLflow UI server"
    echo "  jupyter            Start Jupyter Notebook server"
    echo "  bash               Start bash shell"
    echo "  python             Run Python interpreter"
    echo ""
    echo "Examples:"
    echo "  ./entrypoint.sh train --metric cpu --model-type lstm"
    echo "  ./entrypoint.sh mlflow-ui"
    echo "  ./entrypoint.sh jupyter"
    echo ""
    echo "Container usage:"
    echo "  podman run IMAGE train --metric cpu --model-type lstm"
    echo "  podman run -p 5000:5000 IMAGE mlflow-ui"
    echo "  podman run -p 8888:8888 IMAGE jupyter"
}

# Detect if running in container or locally
IN_CONTAINER=false
if [ -f /.dockerenv ] || [ -f /run/.containerenv ]; then
    IN_CONTAINER=true
fi

# Set base directory
if [ "$IN_CONTAINER" = true ]; then
    BASE_DIR="/app"
else
    # When running locally, use the script's directory
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# Change to base directory
cd "$BASE_DIR"

# Set Python path if not already set
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH="$BASE_DIR"
else
    export PYTHONPATH="$PYTHONPATH:$BASE_DIR"
fi

# Set MLflow tracking URI if not already set
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    export MLFLOW_TRACKING_URI="sqlite:///$BASE_DIR/experiments/mlflow.db"
fi

# Default command
CMD="${1:-bash}"

case "$CMD" in
    train)
        shift
        echo "Starting training from: $BASE_DIR"
        echo "Python path: $PYTHONPATH"
        exec python scripts/train_local.py "$@"
        ;;
    mlflow-ui)
        echo "Starting MLflow UI..."
        echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
        echo "Access at: http://localhost:5000"
        exec mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri "$MLFLOW_TRACKING_URI"
        ;;
    jupyter)
        echo "Starting Jupyter Notebook..."
        if [ "$IN_CONTAINER" = true ]; then
            echo "Access at: http://localhost:8888"
            exec jupyter notebook \
                --ip=0.0.0.0 \
                --port=8888 \
                --no-browser \
                --allow-root \
                --NotebookApp.token="" \
                --NotebookApp.password=""
        else
            echo "Access at: http://localhost:8888"
            # Don't use --allow-root when running locally
            exec jupyter notebook \
                --ip=0.0.0.0 \
                --port=8888 \
                --no-browser
        fi
        ;;
    bash)
        exec /bin/bash
        ;;
    python)
        shift
        exec python "$@"
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        # Execute any other command
        echo "Executing custom command: $*"
        exec "$@"
        ;;
esac
