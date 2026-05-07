import os
import openai

# Load local functions
from import_text import import_articles
from embed import embed
from retrieve import cosine_similarity

# Set the API key
openai.api_key = os.environ["OPENAI_API_KEY"]

ids, texts = import_articles(flatten=True, test=True)
print(ids[:5], texts[:5])