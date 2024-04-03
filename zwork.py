from langchain_community.llms import Ollama

llm = Ollama(model="gemma")

hi = llm.invoke(
    "Tell me python code to sum 4 numbers after asking the user to input it"
)
print(hi)
