#!/bin/bash

# run_benchmark_remote.sh - Execute GeoThought Benchmark on Lambda Cloud
# Usage: ./run_benchmark_remote.sh <model_name> <model_path> [tensor_parallel_size]

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name> <model_path> [tensor_parallel_size]"
    exit 1
fi

MODEL_NAME=$1
MODEL_PATH=$(realpath -m "$2")
TP_SIZE=${3:-1}  # Default TP is 1

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' does not exist."
    echo "Check if you passed the correct path relative to your current directory."
    exit 1
fi

# 1. Unzip Data
if [ ! -d "geometry3k" ]; then
    echo "Unzipping Geometry3k dataset..."
    unzip -q evaluation_script/geometry3k.zip -d evaluation_script/
    if [ $? -ne 0 ]; then
        echo "Error unzipping dataset. Aborting."
        exit 1
    fi
    echo "Dataset unzipped."
else
    echo "Geometry3k dataset found, skipping unzip."
fi

# 2. Start vLLM Server in background
echo "Starting vLLM server for model: $MODEL_NAME (TP=$TP_SIZE)..."
# We need to modify vllm_server.sh or run the command directly here.
# Assuming we use the command directly for better control in this script.

LOG_FILE="vllm_server.log"
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size $TP_SIZE \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --port 8111 \
    > $LOG_FILE 2>&1 &

SERVER_PID=$!
echo "vLLM Server processes started with PID: $SERVER_PID. Logging to $LOG_FILE"

# 3. Wait for Server Readiness
echo "Waiting for server to become ready..."
MAX_RETRIES=30
RETRY_COUNT=0
SERVER_READY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    # Check if the server process is still alive. 
    # If not, no point in waiting further.
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "vLLM Server process (PID: $SERVER_PID) died unexpectedly during startup."
        echo "Check $LOG_FILE for details. Dumping last 20 lines:"
        echo "========================================"
        tail -n 20 $LOG_FILE
        echo "========================================"
        exit 1
    fi

    # Check if it responds to HTTP requests
    curl -s http://localhost:8111/v1/models > /dev/null
    if [ $? -eq 0 ]; then
        SERVER_READY=1
        echo "Server is ready!"
        break
    fi
    echo "Server not ready yet... waiting 10s ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $SERVER_READY -eq 0 ]; then
    echo "Server failed to start within timeout. Check $LOG_FILE for details."
    kill -9 $SERVER_PID 2>/dev/null
    exit 1
fi

# 4. Run Inference
echo "Running inference..."
# Modify run_infer.sh parameters dynamically or call python script
# We can use the existing `monitor_inference.sh` but it hardcodes `run_infer.sh` call.
# Let's call `run_infer.sh` logic directly but with our parameters.

# We need to construct the `monitor_inference.sh` or `run_infer.sh` equivalent call.
# The original `run_infer.sh` calls `inference.py`.

OUTPUT_DIR="./results"
mkdir -p $OUTPUT_DIR
OUTPUT_FILE="$OUTPUT_DIR/${MODEL_NAME}_results.jsonl"

python3 evaluation_script/inference.py \
  --api_url "http://127.0.0.1:8111" \
  --model_name "$MODEL_NAME" \
  --prompt_path "evaluation_script/geometry3k_test_prompts.jsonl" \
  --image_root "evaluation_script/geometry3k" \
  --output_path "$OUTPUT_FILE" \
  --max_workers 32

# 5. Cleanup
echo "Benchmark complete. Stopping server..."
kill $SERVER_PID
echo "Server stopped."

# 6. Summary
echo "Results saved to $OUTPUT_FILE"
