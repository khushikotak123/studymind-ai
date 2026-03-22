import sys
import os
sys.path.append(os.path.dirname(__file__))

from ai_core.embeddings import ingest_pdf, load_vectorstore

pdf_path = r"D:\studymind-ai\data\raw_pdfs\notes.pdf"
index_name = "web_server_notes"

print("=== Starting ingestion ===")
ingest_pdf(pdf_path, index_name)

print("\n=== Testing search ===")
vs = load_vectorstore(index_name)
results = vs.similarity_search("what is PHP?", k=3)

for i, r in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(r.page_content[:300])