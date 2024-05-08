import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_embeddings(texts, model, cache_file='embeddings_cache.json'):
    """Fetch embeddings for a list of texts, with caching."""
    key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=key)

    # Try to load cached embeddings
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            cache = json.load(file)
    else:
        cache = {}

    embeddings = []
    to_fetch = []
    for text in texts:
        if text in cache:
            embeddings.append(cache[text])
        else:
            to_fetch.append(text)

    # Fetch missing embeddings and update the cache
    if to_fetch:
        for text in to_fetch:
            response = client.embeddings.create(input=[text], model=model)
            embedding = response.data[0].embedding
            cache[text] = embedding
            embeddings.append(embedding)

        # Save updated cache
        with open(cache_file, 'w') as file:
            json.dump(cache, file)

    return embeddings
