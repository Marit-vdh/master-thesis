import os
import openai

# Load local functions
from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import retrieve

# Set global_variables
openai.api_key = os.environ["OPENAI_API_KEY"]
embedder = "infly/inf-retriever-v1"

# Get the query from the user
query = "Amsterdam"
embedded_query = embed(
    embedder, input_articles=query, api_key=openai.api_key, write=False
)

# Load the articles from local memory
ids, texts = import_articles(flatten=True, test=True)

# Create embeddings of the articles and query
# embeddings = embed(
#     model="infly/inf-retriever-v1",
#     input_articles=texts[:5],
#     api_key=openai.api_key,
#     write=True,
#     output_file="data.tsv",
# )

embeddings = load_embeddings("data.tsv", ids)

# Retrieve the documents relevant to the query
relevant_documents = retrieve(
    query=embedded_query,
    embeddings=embeddings,
    texts=dict(zip(ids, texts)),
    n_articles=3,
)
