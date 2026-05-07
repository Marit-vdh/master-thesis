import os
import openai

# Load local functions
from import_text import import_articles
from embed import embed, load_embeddings
from retrieve import cosine_similarity

# Set the API key
openai.api_key = os.environ["OPENAI_API_KEY"]

ids, texts = import_articles(flatten=True, test=True)

embeddings = embed(model='infly/inf-retriever-v1', 
                   input_articles=texts[:5], 
                   api_key=openai.api_key, 
                   write=True, 
                   output_file='data.tsv')

# embeddings = load_embeddings('data.json')
# print(embeddings)