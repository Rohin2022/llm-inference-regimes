from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://0.0.0.0:8000/v1"
)

response = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)