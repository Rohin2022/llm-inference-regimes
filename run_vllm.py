import requests
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://0.0.0.0:8000/v1"
)

METRICS_URL = "http://0.0.0.0:8000/metrics"


def get_metric_sum(metric_name):
    """Get the cumulative sum for a vLLM histogram."""
    metrics = requests.get(METRICS_URL).text

    for line in metrics.splitlines():
        if line.startswith(f"{metric_name}_sum"):
            return float(line.split()[-1])

    return 0.0


model = client.models.list().data[0].id

# Metrics before request
prefill_before = get_metric_sum(
    "vllm:request_prefill_time_seconds"
)
decode_before = get_metric_sum(
    "vllm:request_decode_time_seconds"
)

# Run request
response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100
)

# Metrics after request
prefill_after = get_metric_sum(
    "vllm:request_prefill_time_seconds"
)
decode_after = get_metric_sum(
    "vllm:request_decode_time_seconds"
)

# Time spent on this request
prefill_time = prefill_after - prefill_before
decode_time = decode_after - decode_before

print(response.choices[0].message.content)
print()
print(f"Prefill time: {prefill_time:.6f} s")
print(f"Decode time:  {decode_time:.6f} s")
print(f"Total:        {prefill_time + decode_time:.6f} s")