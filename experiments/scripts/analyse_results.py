import asyncio
import os
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.metrics import DiscreteMetric, NumericMetric
from ragas.metrics.collections import (
    ContextPrecision,
    ContextUtilization,
    ContextEntityRecall,
    NoiseSensitivity,
    AnswerRelevancy,
    Faithfulness,
)
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

import json
import pandas as pd

config = RunConfig(timeout=600, max_retries=5)


def analyse_rag_response(prompt="", context=[], text=""):
    scores = {}
    client = AsyncOpenAI(
        api_key=os.environ["NEBUL_API_KEY"],
        base_url="https://api.inference.nebul.io/v1",
    )
    llm = llm_factory("deepseek-ai/DeepSeek-OCR", client=client)
    embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=client)

    try:
        scores["context_utilization"] = (
            ContextUtilization(llm=llm)
            .score(user_input=prompt, response=text, retrieved_contexts=context)
            .value
        )
    except:
        scores["context_utilization"] = None

    try:
        scores["answer_relevancy"] = (
            AnswerRelevancy(llm=llm, embeddings=embeddings)
            .score(user_input=prompt, response=text)
            .value
        )
    except:
        scores["answer_relevancy"] = None

    try:
        scores["faithfulness"] = (
            Faithfulness(llm=llm)
            .score(user_input=prompt, response=text, retrieved_contexts=context)
            .value
        )
    except:
        scores["faithfullness"] = None

    return scores


def get_results_from_file(filename):

    with open(f"../responses/{filename}", "r") as response:
        response_data = json.load(response)

        result = analyse_rag_response(
            prompt=response_data["user_prompt"],
            context=[v for _, v in response_data["articles_used"].items()],
            text=response_data["response"],
        )

        result["user_prompt"] = response_data["user_prompt"]
        result["embedder"] = response_data["embedder"]
        result["generator"] = response_data["generator"]
        result["distance_metric"] = response_data["distance_metric"]
        result["prompt_type"] = response_data["prompt_type"]
    return result


if __name__ == "__main__":
    # output_file = f"../responses/result_eval.json"
    # if os.path.exists(output_file):
    #     os.remove(output_file)

    # results = []

    # for file in os.listdir("../responses"):
    #     if file.startswith("."):
    #         continue

    #     result = get_results_from_file(file)

    #     results.append(result)

    # with open(f"../responses/result_eval.json", "w") as result_file:
    #     json.dump(results, result_file)

    # results_df = pd.DataFrame(results)
    # print(results_df)

    # Evaluate all responses with embedding-based retrieval
    output_file = f"../results/result_eval.json"
    if os.path.exists(output_file):
        os.remove(output_file)

    results = []

    for embedder_folder in os.listdir("../results"):
        if embedder_folder.startswith("."):
            continue

        if embedder_folder == "BM25":
            for generator_folder in os.listdir(f"../results/{embedder_folder}"):

                if generator_folder.startswith("."):
                    continue

                for result_file in os.listdir(
                    f"../results/{embedder_folder}/{generator_folder}"
                ):
                    if result_file.startswith("."):
                        continue

                    result = get_results_from_file(
                        f"../results/{embedder_folder}/{generator_folder}/{result_file}"
                    )
                    print(result)

                    results.append(result)

        for generator_folder in os.listdir(f"../results/{embedder_folder}"):
            if generator_folder.startswith("."):
                continue

            for distance_metric in os.listdir(
                f"../results/{embedder_folder}/{generator_folder}"
            ):
                if distance_metric.startswith("."):
                    continue

                for result_file in os.listdir(
                    f"../results/{embedder_folder}/{generator_folder}/{distance_metric}"
                ):
                    if distance_metric.startswith("."):
                        continue

                    result = get_results_from_file(
                        f"../results/{embedder_folder}/{generator_folder}/{distance_metric}/{result_file}"
                    )
                    print(result)

                    results.append(result)

    with open(f"../responses/result_eval.json", "w") as result_file:
        json.dump(results, result_file)

    results_df = pd.DataFrame(results)
    print(results_df)
