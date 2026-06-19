import asyncio
import os
from openai import OpenAI
from ragas.metrics import DiscreteMetric, NumericMetric
from ragas.llms import llm_factory

client = OpenAI(
    api_key=os.environ["NEBUL_API_KEY"], base_url="https://api.inference.nebul.io/v1"
)
llm = llm_factory("Qwen/Qwen3-30B-A3B-Instruct-2507", client=client)

# Create a custom aspect evaluator
metric = NumericMetric(
    name="relevance",
    prompt="Rate relevance 1-5: {response} for question: {question}",
    allowed_values=(1, 5),
)


# Score your application's output
async def main():
    score = await metric.ascore(llm=llm, response="The summary of the text is...")
    print(f"Score: {score.value}")  # 'accurate' or 'inaccurate'
    print(f"Reason: {score.reason}")


if __name__ == "__main__":
    asyncio.run(main())

# def evaluate_response():
#     pass
