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

class ScrapeTool:
    def __init__(self, cache: Cache):
        self.cache = cache

    def scrape(self, url: str) -> str:
        cache_key = f"scrape:{url}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
            response = requests.get(url, headers=headers, timeout=10)
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

class GroqTool:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def process_job_listing(self, text: str) -> dict:
        response = self.call_groq_api(text)
        return {
            "company": response.get("company", "N/A"),
            "job_role": response.get("job_role", "N/A"),
            "salary": response.get("salary", "N/A"),
            "requirements": response.get("requirements", "N/A"),
            "link": response.get("link", "N/A")
        }

    def call_groq_api(self, text: str) -> dict:
        # Placeholder function for calling Groq API
        # Replace this with actual API call and response parsing
        return {
            "company": "Example Company",
            "job_role": "Example Job Role",
            "salary": "Example Salary",
            "requirements": "Example Requirements",
            "link": "http://example.com/job-link"
        }

# Initialize the cache
cache = Cache()

# Create instances of the tools
search_tool = SearchTool(cache)
scrape_tool = ScrapeTool(cache)
groq_tool = GroqTool(api_key="")  # Replace with your actual Groq API key

# Example usage
query = "Django developer jobs in Germany"
search_results = search_tool.search(query)
print("the jobs are at the following link", search_results)

for url in search_results:
    scraped_text = scrape_tool.scrape(url)
    job_info = groq_tool.process_job_listing(scraped_text)
    print(f"Company: {job_info['company']}")
    print(f"Job Role: {job_info['job_role']}")
    print(f"Salary: {job_info['salary']}")
    print(f"Requirements: {job_info['requirements']}")
    print(f"Link: {job_info['link']}")
