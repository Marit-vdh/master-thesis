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
    FactualCorrectness,
    BleuScore,
)
from ragas.llms import llm_factory
from ragas.run_config import RunConfig

import json
import pandas as pd

config = RunConfig(timeout=600, max_retries=5)


def analyse_rag_response(prompt="", context=[], reference="", text=""):
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


def analyse_with_reference(
    prompt="", reference="", response="", answer_relevancy=False
):
    scores = {}
    client = AsyncOpenAI(
        api_key=os.environ["NEBUL_API_KEY"],
        base_url="https://api.inference.nebul.io/v1",
    )
    llm = llm_factory("deepseek-ai/DeepSeek-OCR", client=client)
    embeddings = embedding_factory("openai", model="BAAI/bge-m3", client=client)

    if answer_relevancy:
        try:
            scores["answer_relevancy"] = (
                AnswerRelevancy(llm=llm, embeddings=embeddings)
                .score(user_input=prompt, response=response)
                .value
            )
        except:
            scores["answer_relevancy"] = None

    try:
        scores["factual_correctness"] = (
            FactualCorrectness(llm=llm)
            .score(response=response, reference=reference)
            .value
        )

    except:
        scores["factual_correctness"] = None

    try:
        scores["bleu"] = BleuScore().score(response=response, reference=reference).value
    except:
        scores["bleu"] = None

    return scores


def get_results_from_file(filename, references=[]):
    i = int(filename.split("_")[-2])

    with open(f"../responses/{filename}", "r") as response:

        response_data = json.load(response)

        result = analyse_rag_response(
            prompt=response_data["user_prompt"],
            context=[v for _, v in response_data["articles_used"].items()],
            text=response_data["response"],
        )

        result = result | analyse_with_reference(
            prompt=response_data["user_prompt"],
            reference=references[i],
            response=response_data["response"],
        )

        result["user_prompt"] = response_data["user_prompt"]
        result["embedder"] = response_data["embedder"]
        result["generator"] = response_data["generator"]
        result["distance_metric"] = response_data["distance_metric"]
        result["prompt_type"] = response_data["prompt_type"]
    return result


if __name__ == "__main__":
    # print(
    #     get_results_from_file(
    #         "../results/BM25/Ministral-3-14B-Instruct-2512/prompt_0_response.json",
    #         [
    #             "Hoofdstad van Nederland; grootste stad in Noord-Holland,centrum voor cultuur,economie en toerisme."
    #         ],
    #     )
    # )

    # Evaluate all responses

    # Load the references from the reference file
    references = []
    with open("../sources/references.txt", "r") as file:
        for line in file:
            references.append(line.strip())

    output_file = f"../results/result_eval.json"
    if os.path.exists(output_file):
        os.remove(output_file)

    results = []

    for embedder_folder in os.listdir("../results"):
        if embedder_folder.startswith(".") or embedder_folder == "generation_only":
            continue

        elif embedder_folder == "BM25":
            for generator_folder in os.listdir(f"../results/{embedder_folder}"):

                if generator_folder.startswith("."):
                    continue

                for result_file in os.listdir(
                    f"../results/{embedder_folder}/{generator_folder}"
                ):
                    if result_file.startswith("."):
                        continue

                    result = get_results_from_file(
                        f"../results/{embedder_folder}/{generator_folder}/{result_file}",
                        references=references,
                    )
                    print(result)

                    results.append(result)

        else:
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
                            f"../results/{embedder_folder}/{generator_folder}/{distance_metric}/{result_file}",
                            references=references,
                        )
                        print(result)

                        results.append(result)

    with open(f"../results/result_eval.json", "w") as result_file:
        json.dump(results, result_file)

    results_df = pd.DataFrame(results)
    print(results_df)
