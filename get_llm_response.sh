#!/bin/bash
set -e

export NO_PROXY="localhost,127.0.0.1,::1,$NO_PROXY"
export no_proxy="$NO_PROXY"

# Start vLLM
vllm serve ./HFCache/hub/models--allenai--OLMo-2-0425-1B-Instruct/snapshots/48d788eca847d4d7548f375ad03d3c9312f6139e \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization fp8 \
    --kv-cache-dtype fp8 > vllm.log 2>&1 &
    

VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM..."

    # Kill vLLM and its children
    pkill -TERM -P "$VLLM_PID" 2>/dev/null || true
    kill "$VLLM_PID" 2>/dev/null || true

    sleep 3

    # Force kill anything still alive
    pkill -KILL -P "$VLLM_PID" 2>/dev/null || true
    kill -9 "$VLLM_PID" 2>/dev/null || true

    echo "vLLM stopped."
}

trap cleanup EXIT INT TERM

echo "Waiting for vLLM..."

until curl -sf http://localhost:8000/v1/models > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM crashed. Check vllm.log:"
        cat vllm.log
        exit 1
    fi
    sleep 1
done

echo "vLLM is ready!"

python run_vllm.py