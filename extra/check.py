import requests
from bs4 import BeautifulSoup
from typing import List
from googlesearch import search
import json
import logging
from itertools import islice
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
)

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
        if (cached_result):
            return cached_result
        try:
            result = list(islice(search(query), num_results))
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"Error during search: {str(e)}")
            return []

def load_data(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

def scrape_content(url: str) -> str:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logging.error(f"Error scraping {url}: {str(e)}")
        return ""

# Initialize the cache
cache = Cache()

# Create instances of the tools
search_tool = SearchTool(cache)

# Example usage
query = "Django developer jobs in Germany"
search_results = search_tool.search(query)
print("The jobs are at the following links:", search_results)

# Scrape the content from the URLs
scraped_content = [load_data(search_results) for url in search_results]  # load_data(search_results)
print("Scraped content:", scraped_content)

# Load the data using LangChain WebBaseLoader


# Setup LangChain with a prompt and an LLM (e.g., OpenAI)

prompt = PromptTemplate(input_variables=["content"], template="Analyze the following content: {content}")

chain = LLMChain(llm=llm, prompt=prompt)

# Process each piece of content with the LLM
for content in scraped_content:
    result = chain.run(content)
    print("LLM Analysis:", result)
