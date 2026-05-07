import json
import requests


def embed(
    model: str,
    input_articles: list[str],
    api_key: str,
    write: bool = True,
    output_file: str = "data.tsv",
) -> list[list[float]]:
    """Embed a list of strings using the Nebul platform.

    Args:
        model (str): the model used for embedding. Has to be a model that is available through Nebul.
        input_articles (list[str]): strings representing the text of articles.
        api_key (str): API key for access to the Nebul platform.
        write (bool, optional): set to true to write to a local file. Defaults to True.
        output_file (str, optional): sets the name of the output file in case write is set to true. Defaults to "data.tsv".

    Returns:
        list[list[float]]: a list containing the embeddings, each represented as a list.
    """
    print("Start embedding")

    url = "https://api.inference.nebul.io/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {"model": model, "input": input_articles, "normalize": True}

    print("Request embeddings")
    response = requests.post(url, headers=headers, json=payload)

    print("Load embeddings as dictionary")
    data = response.json()["data"]

    embeddings = []
    indices = []

    for article in data:
        embeddings += [article["embedding"]]
        indices += [article["index"]]

    if write:
        print("Writing embeddings to output file")
        with open(output_file, "w") as output_file:
            for i in range(len(embeddings)):
                output_file.write(f"{indices[i]}\t{embeddings[i]}\n")

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
