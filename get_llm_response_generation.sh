#!/bin/bash
set -e

export NO_PROXY="localhost,127.0.0.1,::1,$NO_PROXY"
export no_proxy="$NO_PROXY"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

# Start vLLM
vllm serve /groups/bodymaps/Rohin/LMAgents/HFCache/hub/models--allenai--OLMoE-1B-7B-0924-Instruct/snapshots/7f1c97f440f06ce36705e4f2b843edb5925f4498 \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization fp8 \
    --kv-cache-dtype fp8 > vllm_generation.log 2>&1 &
    

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

python run_vllm.py \
    --csv_path crs_reports_sample_fits_olmo_window.csv \
    --output_path results/models--allenai--OLMoE-1B-7B-0924-Instruct/generation.csv \
    --llm_output_path results/models--allenai--OLMoE-1B-7B-0924-Instruct/generation_outputs.jsonl \
    --task CREATION \
    --max_tokens 3000