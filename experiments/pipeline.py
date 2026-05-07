import os
import openai

# Load local functions
from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import retrieve
from generator import generate_response

# Set global_variables
openai.api_key = os.environ["OPENAI_API_KEY"]
embedder = "infly/inf-retriever-v1"
generator = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Get the query from the user
user_prompt = "Amsterdam"
embedded_prompt = embed(
    embedder, input_articles=user_prompt, api_key=openai.api_key, write=False
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
    query=embedded_prompt,
    embeddings=embeddings,
    texts=dict(zip(ids, texts)),
    n_articles=3,
)

# Set up de system prompt using the relevant articles
system_prompt = f"You are a helpful chatbot. Use only the following pieces of context to answer the question. Don't make up any new information: {'\n'.join([f' - {chunk}' for chunk in relevant_documents])}"

# Generate a response
response = generate_response(
    api_key=openai.api_key,
    model=generator,
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

print(response)


def main():
    run_single_experiment()
