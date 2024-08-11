import requests
from bs4 import BeautifulSoup
from typing import List
from googlesearch import search
import json
import logging
from itertools import islice

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class Cache:
    def __init__(self, cache_file="cache.json"):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self.save_cache()

class SearchTool:
    def __init__(self, cache: Cache):
        self.cache = cache

    def search(self, query: str, num_results: int = 5) -> List[str]:
        cache_key = f"search:{query}:{num_results}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            result = list(islice(search(query), num_results))
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []


# Initialize the cache
cache = Cache()

# Create instances of the tools
search_tool = SearchTool(cache)



# Example usage
query = "Django developer jobs in Germany"
search_results = search_tool.search(query)
print("the jobs are at the following link", search_results)

from langchain_community.document_loaders import WebBaseLoader

def load_data(urls: List[str]):
    loader = WebBaseLoader(urls)
    data = loader.load()
    return data



hi = load_data(urls)
print(hi)