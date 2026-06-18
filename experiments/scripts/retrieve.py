import numpy as np
from math import dist
from tqdm import tqdm
import json
import bm25s
import Stemmer


def create_index(
    ids=None,
    texts=None,
    index_dir="../sources/bm25_index",
    verbose=True,
    load_new_index=False,
):
    if load_new_index:

        if ids == None or texts == None:

            ids, texts = import_text.import_articles(flatten=True)

        corpus_records = [{"id": k, "text": v} for k, v in zip(ids, texts)]

        print("Creating index") if verbose else None

        # Create vocabulary of the texts in the corpus
        tokenizer = bm25s.tokenization.Tokenizer()
        corpus_tokens = tokenizer.tokenize(texts, return_as="tuple")

        # Initiate retriever
        retriever = bm25s.BM25(corpus=corpus_records, backend="numba")
        retriever.index(corpus_tokens)

        # Save retriever, vocabulary and stopwords for future use
        retriever.save(index_dir)
        tokenizer.save_vocab(index_dir)
        tokenizer.save_stopwords(index_dir)

        print("Finished creating index") if verbose else None

        print(f"Saved the index to {index_dir}.") if verbose else None

    else:
        print("Loading index") if verbose else None

        # Load retriever and tokenizer from files
        retriever = bm25s.BM25.load(f"{index_dir}", load_corpus=True)
        tokenizer = bm25s.tokenization.Tokenizer()
        tokenizer.load_stopwords(save_dir=index_dir)
        tokenizer.load_vocab(save_dir=index_dir)

        print("Finished loading index") if verbose else None

    return retriever, tokenizer


def bm25_retriever(retriever, tokenizer, queries, n_articles=3, print_result=False):

    queries_tokenized = tokenizer.tokenize(queries)

    # Retrieve the top-k results
    results = retriever.retrieve(queries_tokenized, k=n_articles)

    # show first results if desired
    docs = []
    ids = []
    for i in range(len(queries)):
        print(queries[i]) if print_result else None
        for j in range(n_articles):
            (
                print(
                    f'{results.scores[i, j]} - {results.documents[i, j]["id"]} - {results.documents[i, j]["text"]}'
                )
                if print_result
                else None
            )
            docs.append(results.documents[i, j]["text"])
            ids.append(results.documents[i, j]["id"])

    return docs, ids


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


def retrieve_vector(
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

            split_line = line.split("\t")

            index = int(split_line[0])
            vector = json.loads(split_line[1])

            # Map the index back to the article name provided in 'indices'
            article_name = indices[index]

            similarities[article_name] = retriever(query, np.array(vector))

    top_embeddings = sorted(similarities, key=similarities.get, reverse=True)

    return [texts[text_id] for text_id in top_embeddings[:n_articles]], top_embeddings[
        :n_articles
    ]


if __name__ == "__main__":
    import import_text

    index, tokenizer = create_index(load_new_index=False)

    bm25_retriever(
        index,
        tokenizer=tokenizer,
        queries=["Amsterdam", "Trekkertrek", "Bruinvis"],
        k=3,
        print_result=True,
    )
