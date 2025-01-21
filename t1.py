'''
# Use async methods to implement:
# and use llm to analyze the stock data
'''

import asyncio

from openai import AsyncOpenAI

conf = {
    "openai": (
        "https://api.openai.com/v1",
        # Replace with your own API key
        "",
        "chatgpt-4o-latest"
    )
}

clients = {}
for name, (base_url, api_key, model) in conf.items():
    clients[name] = AsyncOpenAI(base_url=base_url, api_key=api_key)

async def call_llm(name: str, prompt: str):
    if name not in clients:
        raise ValueError(f"Client {name} not found")
    client = clients[name]
    _, _, model = conf[name]  # Get model from conf tuple
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def test_llm():
    prompt = "What is the meaning of life?"
    result = await call_llm("openai", prompt)
    print(result)

if __name__ == "__main__":
    asyncio.run(test_llm())