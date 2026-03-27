import os
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List

load_dotenv()

VECTOR_STORE_PATH = "/tmp/vector_store"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
EMBEDDING_DIM = 384


class HFEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        results = self._embed_batch([text])
        return results[0]

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            raise RuntimeError(
                "HF_TOKEN environment secret is not set. "
                "Please add it in the Replit Secrets panel."
            )

        truncated = [t[:512] for t in texts]

        response = requests.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": truncated},
            timeout=60,
        )

        if response.status_code == 401:
            raise RuntimeError(
                "HuggingFace API returned 401 Unauthorized. "
                "Check that your HF_TOKEN secret is correct."
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace API error {response.status_code}: {response.text[:300]}"
            )

        data = response.json()

        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected embedding response format: {type(data)}")

        results = []
        for item in data:
            if isinstance(item, list) and len(item) > 0:
                if isinstance(item[0], list):
                    results.append(item[0])
                else:
                    results.append(item)
            else:
                raise RuntimeError(f"Unexpected embedding item format: {item}")

        return results


def get_embeddings():
    return HFEmbeddings()


def ingest_pdf(pdf_path: str, index_name: str):
    print(f"Loading PDF: {pdf_path}")

    import pdfplumber
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    print(f"Extracted {len(text)} characters")

    if not text.strip():
        raise ValueError("Could not extract text from PDF. The file may be scanned or image-based.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.create_documents([text])
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_path = os.path.join(VECTOR_STORE_PATH, index_name)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)
    print(f"Saved vector store to {save_path}")
    return vectorstore


def load_vectorstore(index_name: str):
    embeddings = get_embeddings()
    save_path = os.path.join(VECTOR_STORE_PATH, index_name)
    if not os.path.exists(save_path):
        raise FileNotFoundError(
            f"No vector store found for '{index_name}'. "
            "Please upload and process a PDF first."
        )
    return FAISS.load_local(
        save_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
