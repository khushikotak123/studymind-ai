import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ai_core.embeddings import load_vectorstore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

def generate_quiz(index_name: str, topic: str, num_questions: int = 3):
    vectorstore = load_vectorstore(index_name)
    docs = vectorstore.similarity_search(topic, k=4)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.5
    )

    prompt = PromptTemplate.from_template("""
You are a college exam question generator.
Based on the context below, generate exactly {num_questions} multiple choice questions.

Context:
{context}

Topic: {topic}

Return ONLY a valid JSON array, no extra text, in this exact format:
[
  {{
    "question": "question text here",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1",
    "explanation": "brief explanation why"
  }}
]
""")

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "context": context,
        "topic": topic,
        "num_questions": num_questions
    })

    try:
        start = result.find("[")
        end = result.rfind("]") + 1
        json_str = result[start:end]
        questions = json.loads(json_str)
        return questions
    except Exception as e:
        print(f"Parse error: {e}")
        print(f"Raw output: {result}")
        return []