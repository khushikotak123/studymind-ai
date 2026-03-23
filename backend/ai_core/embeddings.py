import os
import requests
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH =  "/tmp/vector_store"

class GroqEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        response = requests.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={"model": "nomic-embed-text-v1.5", "input": text}
        )
        return response.json()["data"][0]["embedding"]

def get_embeddings():
    return GroqEmbeddings()

def ingest_pdf(pdf_path: str, index_name: str):
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_path = os.path.join(VECTOR_STORE_PATH, index_name)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Saved to {save_path}")
    return vectorstore

def load_vectorstore(index_name: str):
    embeddings = get_embeddings()
    save_path = os.path.join(VECTOR_STORE_PATH, index_name)
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True
    )