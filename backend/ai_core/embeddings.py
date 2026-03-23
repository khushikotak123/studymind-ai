import os
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()

VECTOR_STORE_PATH = "/tmp/vector_store"

class HFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            embedding = self._embed(text)
            results.append(embedding)
        return results

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        try:
            HF_TOKEN = os.getenv("HF_TOKEN")
            print(f"HF_TOKEN exists: {bool(HF_TOKEN)}")
            response = requests.post(
                "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": text[:512]},
                timeout=30
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            data = response.json()
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], list):
                    return data[0]
                if isinstance(data[0], float):
                    return data
            print(f"Unexpected format: {type(data)}")
            return [0.0] * 384
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 384

def get_embeddings():
    return HFEmbeddings()

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