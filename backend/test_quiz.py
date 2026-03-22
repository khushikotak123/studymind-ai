import sys
import os
sys.path.append(os.path.dirname(__file__))

from ai_core.quiz_agent import generate_quiz

index_name = "web_server_notes"
topic = "PHP"

print("=== StudyMind AI - Quiz Generator ===\n")
questions = generate_quiz(index_name, topic, num_questions=3)

for i, q in enumerate(questions):
    print(f"Q{i+1}: {q['question']}")
    for opt in q['options']:
        print(f"     {opt}")
    print(f"Answer: {q['answer']}")
    print(f"Explanation: {q['explanation']}")
    print("-" * 50)