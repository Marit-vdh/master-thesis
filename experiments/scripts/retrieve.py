import numpy as np
from math import dist
from tqdm import tqdm
import json


def cosine_similarity(a, b):
    """Compute the cosine similarity between two arrays. A cosine similarity of 1
    indicates a vector that has the same angle, 0 indicates
    orthogonal vectors and -1 point in opposite directions.

    Args:
        a (np.array): array representing the embedding of a paragraph of text.
        b (np.array): array representing the embedding of a prompt.

    Returns:
        (float): represents the degree of which two vectors point in the same
        direction, creating a measure of similarity on an interval of -1 and 1.
    """
    # Compute the dot product
    dot_product = np.dot(a, b)

    # Compute the magnitudes of the vectors
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Calculate cosine similarity
    return (dot_product / (magnitude_a * magnitude_b))[0]


def dot(a, b):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
    """
    return -np.dot(a, b)


def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two arrays, which is the square
    root of the sum of squares of the elements of the arrays.

    Args:
        a (np.array): array representing the embedding of a paragraph of text.
        b (np.array): array representing the embedding of a prompt.

    Returns:
        (float): Represents the distance between two points.
    """
    return dist(a, b)


def manhattan(a, b):
    return np.sum(abs(a - b))


def retrieve(
    retriever: function,
    query: list[float],
    embeddings_file,
    indices,
    # embeddings: dict[str, list[float]],
    texts: dict[str, str],
    n_articles: int = 3,
    verbose=False,
) -> list[str]:
    """Retrieve the top matching documents based on the given retrieval method.

    Args:
        retriever (function): retrieval function that works based on embeddings.
        query (list[float]): embedded user prompt.
        embeddings (dict[str, list[float]]): array of text embeddings to match the query with.
        texts (dict[str, str]): original texts to return a readable output.
        n_articles (int): number of articles to return.

    Returns:
        list[str]: list of articles ordered on similarity based on the retrieval method.
    """

    query = np.array(query)

    similarities = {}

    print("Calculating similarities") if verbose else None

    with open(embeddings_file) as embeddings:

        for line in embeddings:

            try:
                split_line = line.split("\t")

                if len(split_line) < 2:
                    print(f"DEBUG WARNING: Skipping malformed line")
                    continue

                index = int(split_line[0])
                vector = json.loads(split_line[1])

                # Map the index back to the article name provided in 'indices'
                article_name = indices[index]

                similarities[article_name] = retriever(query, np.array(vector))

            except (ValueError, IndexError, json.JSONDecodeError) as e:
                print(f"DEBUG ERROR: Line {index} failed to parse: {e}")

    # for index, embedding in embeddings.items():

    #     similarities[index] = retriever(query, np.array(embedding))

    top_embeddings = sorted(similarities, key=similarities.get, reverse=True)

    return [texts[text_id] for text_id in top_embeddings[:n_articles]], top_embeddings[
        :n_articles
    ]
