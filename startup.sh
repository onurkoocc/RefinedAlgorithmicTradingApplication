#!/bin/bash
set -e

echo "Starting Bitcoin Trading System..."

# Create necessary directories
mkdir -p data results/models results/backtest results/logs

# Remove the config file check
echo "Using default configuration from Config class."

# Check for NVIDIA GPU and provide more detailed information
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    echo "NVIDIA GPU detected: $GPU_INFO"
else
    echo "WARNING: NVIDIA GPU not detected. Performance will be significantly reduced."
    echo "CPU will be used for model training and inference."
fi

# Check for API credentials
if [ -n "$BINANCE_API_KEY" ] && [ -n "$BINANCE_API_SECRET" ]; then
    echo "Binance API credentials detected."
    API_FLAG="--use-api"
else
    echo "No Binance API credentials found. Using cached data."
    API_FLAG=""
fi

# Run the application with proper logging
echo "Running backtesting..."
python main.py --mode backtest $API_FLAG "$@"
#python main.py --mode optimize-features --optuna-trials 50 $API_FLAG "$@"
# Check exit code
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Error: Application exited with code $exit_code"
    exit $exit_code
fi

echo "Process completed successfully."