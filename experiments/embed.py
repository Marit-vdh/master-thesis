import json


def embed(client, model, input_articles, write=True, output_file="data.json"):
    """Embed an array of input articles.

    Args:
        client (_type_): _description_
        model (_type_): _description_
        input_articles (_type_): _description_
        write (bool, optional): _description_. Defaults to True.
        output_file (str, optional): _description_. Defaults to 'data.json'.

    Returns:
        _type_: _description_
    """
    response = client.embeddings.create(model=model, input=input_articles)
    data = response.json()

    if write:
        with open(output_file, "w") as json_file:
            json.dump(data, json_file)

    return data
