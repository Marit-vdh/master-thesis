import os
import json
import requests
from tqdm import tqdm
from time import sleep
from random import uniform


def embed(
    model: str,
    input_articles: list[str],
    api_key: str,
    write: bool = True,
    output_file: str = "data.tsv",
    verbose: bool = True,
    max_articles: int = 5,
) -> list[list[float]]:

    url = "https://api.inference.nebul.io/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if write and os.path.exists(output_file):
        os.remove(output_file)

    all_embeddings = []

    for i in range(0, len(input_articles), max_articles):
        for _ in range(4):
            try:
                end = min(i + max_articles, len(input_articles))
                raw_partition = input_articles[i:end]

                # Truncate articles that are too long
                article_partition = [text[:35000] for text in raw_partition]

                payload = {
                    "model": model,
                    "input": article_partition,
                    "normalize": True,
                    "embed_dtype": "float16",
                }

                response = requests.post(
                    url, headers=headers, json=payload, timeout=300
                )

                if response.status_code != 200:
                    if verbose:
                        print(f"DEBUG ERROR: {response.status_code} - {response.text}")
                    response.raise_for_status()

                data = response.json()["data"]

                batch_embeddings = [article["embedding"] for article in data]
                batch_indices = [int(article["index"]) + i for article in data]

                # Append batch results to the full list
                all_embeddings.extend(batch_embeddings)

                if write:
                    with open(output_file, "a") as data_file:
                        for idx, emb in zip(batch_indices, batch_embeddings):
                            data_file.write(f"{idx}\t{json.dumps(emb)}\n")

                sleep(uniform(0.5, 1.5))

                str_error = None

            except Exception as e:
                str_error = e
                if verbose:
                    print(f"DEBUG: Batch {i} fail: {e}")
                sleep(4)

            if not str_error:
                break

    return all_embeddings


def load_embeddings(
    embedding_file: str, indices: list[str], verbose=False
) -> dict[str, list[float]]:
    """Load embeddings with validation checks."""
    embedding_dict = {}

    if not os.path.exists(embedding_file):
        print(f"DEBUG ERROR: File {embedding_file} not found.")
        return {}

    with open(embedding_file, "r") as f:
        lines = f.readlines()

    if verbose:
        print(f"DEBUG: Loading {len(lines)} lines from {embedding_file}")

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue  # Skip empty lines

        try:
            split_line = line.split("\t")
            if len(split_line) < 2:
                print(f"DEBUG WARNING: Skipping malformed line {line_num}")
                continue

            idx = int(split_line[0])
            vector = json.loads(split_line[1])

            # Map the index back to the article name provided in 'indices'
            article_name = indices[idx]
            embedding_dict[article_name] = vector

        except (ValueError, IndexError, json.JSONDecodeError) as e:
            print(f"DEBUG ERROR: Line {line_num} failed to parse: {e}")

    if verbose:
        print(f"DEBUG: Successfully loaded {len(embedding_dict)} embeddings.")

    return embedding_dict
