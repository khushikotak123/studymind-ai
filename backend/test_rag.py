import sys
import os
sys.path.append(os.path.dirname(__file__))

from ai_core.rag_pipeline import ask_question

index_name = "web_server_notes"

print("=== StudyMind AI - RAG Test ===\n")

questions = [
    "What is PHP?",
    "What is the difference between Apache and Nginx?",
    "What are PHP magic constants?"
]

for q in questions:
    print(f"Q: {q}")
    answer = ask_question(index_name, q)
    print(f"A: {answer}")
    print("-" * 50)