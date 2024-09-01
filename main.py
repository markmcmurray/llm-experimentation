import time
import warnings
import os

import ollama

warnings.filterwarnings("ignore")
from pprint import pprint

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


loader = GenericLoader.from_filesystem(
    "/Users/markmcmurray/workspace/forks/helm-issue-13241",
    glob="pkg/cli/**/*",
    suffixes=[".go"],
    parser=LanguageParser(language=Language.GO),
)
docs = loader.load()
print(len(docs))
go_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.GO, chunk_size=60, chunk_overlap=0
)

chunks = go_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(
    model="deepseek-coder-v2")

faiss_index_path = "faiss_index.faiss"

max_retries = 5
retry_delay = 3  # seconds

if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
else:
    chunk_texts = [chunk.page_content for chunk in chunks]
    for attempt in range(max_retries):
        try:
            chunk_embeddings = embeddings.embed_documents(chunk_texts)
            print(len(chunk_texts))
            print(len(chunk_embeddings))
            code_embeddings = list(zip(chunk_texts, chunk_embeddings))
            break
        except ollama._types.ResponseError as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                raise  # Re-raise the last exception
    vectorstore = FAISS.from_embeddings(code_embeddings, embeddings)
    vectorstore.save_local(faiss_index_path)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("Find tests")

print(docs)

# llm = OllamaLLM(model="deepseek-coder-v2")
# qa_chain = RetrievalQA(llm=llm, retriever=vectorstore.as_retriever())

# question = "How does the main function work in this repository?"
# answer = qa_chain.run(question)

# print(answer)