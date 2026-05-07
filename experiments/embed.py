import json
import requests
import os

def embed(model, input_articles, api_key, write=True, output_file="data.tsv"):
    """Embed an array of input articles.

    Args:
        client (_type_): _description_
        model (_type_): _description_
        input_articles (_type_): _description_
        write (bool, optional): _description_. Defaults to True.
        output_file (str, optional): _description_. Defaults to 'data.tsv'.

    Returns:
        _type_: _description_
    """
    print("Start embedding")

    url = "https://api.inference.nebul.io/v1/embeddings"
    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    }

    payload = {
    "model": model,
    "input": input_articles,
    "normalize" : True
    }

    print("Request embeddings")
    response = requests.post(url, headers=headers, json=payload)

    print("Load embeddings as dictionary")
    data = response.json()['data']

    embeddings = []
    indices = []

    for article in data:
        embeddings += [article['embedding']]
        indices += [article['index']]

    if write:
        print("Writing embeddings to output file")
        with open(output_file, "w") as output_file:
            for i in range(len(embeddings)):
                output_file.write(f'{indices[i]}\t{embeddings[i]}\n')

    return embeddings

def load_embeddings(embedding_file):
    if os.path.exists(embedding_file):
        embedding_dict = json.load(embedding_file)
    return embedding_dict
