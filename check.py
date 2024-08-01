import requests
from bs4 import BeautifulSoup
from typing import List
from googlesearch import search  # Ensure you have this installed: pip install googlesearch-python
import json
import logging
from itertools import islice

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
            # Use islice to limit the number of search results
            result = list(islice(search(query), num_results))
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

class ScrapeTool:
    def __init__(self, cache: Cache):
        self.cache = cache

    def scrape(self, url: str) -> str:
        cache_key = f"scrape:{url}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3'])])
            result = text[:1000]
            self.cache.set(cache_key, result)
            return result
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error scraping {url}: {str(e)}")
            return f"Error scraping {url}: {str(e)}"
        except Exception as e:
            logging.error(f"Error scraping {url}: {str(e)}")
            return f"Error scraping {url}: {str(e)}"

# Initialize the cache
cache = Cache()

# Create instances of the tools
search_tool = SearchTool(cache)
scrape_tool = ScrapeTool(cache)

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
query = "things to visit in germany"
search_results = search_tool.search(query)
print(search_results)
for url in search_results:
    print(scrape_tool.scrape(url))
