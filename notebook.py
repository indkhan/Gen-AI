from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

loader = TextLoader("./data/acme_bank_faq.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)
print("- loaded bank docs")
vector_store = FAISS.from_documents(splits, embeddings)
print("- created bank index")

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)
result = qa_chain.invoke(
    {"question": "How do I unblock my credit card?", "chat_history": []}
)
print(result["answer"])
