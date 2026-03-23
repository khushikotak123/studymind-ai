import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "../../data/vector_store")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

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
```

Actually the real problem is `torch` — it's 2GB alone. Let's switch to a Railway-friendly setup. Add this variable in Railway Variables tab:
```
TRANSFORMERS_OFFLINE=1