import os
from openai import OpenAI

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI(base_url="https://api.inference.nebul.io/v1")

response = client.chat.completions.create(
    model="mistralai/Ministral-3-14B-Instruct-2512",
    messages=[{"role": "user", "content": "Write a haiku about AI."}],
)

print(response.choices[0].message.content)
