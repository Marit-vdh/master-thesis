import os
import json
import requests


def embed(
    model: str,
    input_articles: list[str],
    api_key: str,
    write: bool = True,
    output_file: str = "data.tsv",
    verbose: bool = True,
    max_articles: int = 10,
) -> list[list[float]]:
    """Embed a list of strings using the Nebul platform.
    Nebul has a maximum number of embeddings it can process at once, so the embedding has to
    be done in partitions.

    Args:
        model (str): the model used for embedding. Has to be a model that is available through Nebul.
        input_articles (list[str]): strings representing the text of articles.
        api_key (str): API key for access to the Nebul platform.
        write (bool, optional): set to true to write to a local file. Defaults to True.
        output_file (str, optional): sets the name of the output file in case write is set to true. Defaults to "data.tsv".
        verbose (bool, optional): allows for optional reporting of intermediate status.

    Returns:
        (list[list[float]]): a list containing the embeddings, each represented as a list.
    """
    print("Start embedding") if verbose else None

    url = "https://api.inference.nebul.io/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if write:
        if os.path.exists(output_file):
            os.remove(output_file)

    for i in range(0, len(input_articles), max_articles):

        end = (
            i + max_articles
            if i + max_articles < len(input_articles)
            else len(input_articles)
        )

        article_partition = input_articles[i:end]

        payload = {"model": model, "input": article_partition, "normalize": True}

        print("Request embeddings") if verbose else None
        response = requests.post(url, headers=headers, json=payload)

        print("Load embeddings as dictionary") if verbose else None

        data = response.json()["data"]

        embeddings = []
        indices = []

        for article in data:
            embeddings += [article["embedding"]]
            indices += [int(article["index"]) + i]

        print(len(embeddings))

        if write:
            print("Writing embeddings to output file") if verbose else None
            with open(output_file, "a") as data_file:
                for i in range(len(embeddings)):
                    data_file.write(f"{indices[i]}\t{embeddings[i]}\n")

    return embeddings


def load_embeddings(embedding_file: str, indices: list[str]) -> dict[str, list[float]]:
    """

    Args:
        embedding_file (str): name of the file storing the embeddings
        indices (list[str]): article names

    Returns:
        tuple[list[float], dict[str, list[float]]]: embedded query and texts
    """

    embedding_dict = {}
    with open(embedding_file, "r") as embeddings:

        # Store the embeddings in a dictionary where the key is the article name
        # and the value is the embedding, creating a matching dictionary to the
        # unflattened version in import_text
        i = 0

        for line in embeddings:
            split_line = line.split("\t")
            embedding_dict[indices[i]] = json.loads(split_line[1])
            i += 1

    return embedding_dict
