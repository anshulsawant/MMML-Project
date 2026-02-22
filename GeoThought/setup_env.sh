#!/bin/bash

# setup_env.sh - Setup environment for GeoThought Benchmark on Lambda Cloud

echo "Starting environment setup..."

# 1. Install system dependencies (usually not needed on Lambda but good to have)
sudo apt-get update && sudo apt-get install -y git unzip wget

# 2. Install Python dependencies
# We use the system python or the one provided by the image (usually conda or py3.10+)
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install vllm deepspeed pandas tqdm requests pillow huggingface_hub

echo "Environment setup complete."
