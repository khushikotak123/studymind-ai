from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_core.embeddings import ingest_pdf
from ai_core.rag_pipeline import ask_question
from ai_core.quiz_agent import generate_quiz

app = FastAPI(title="StudyMind AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw_pdfs")

class QuestionRequest(BaseModel):
    index_name: str
    question: str

class QuizRequest(BaseModel):
    index_name: str
    topic: str
    num_questions: int = 3

@app.get("/")
def root():
    return {"message": "StudyMind AI is running!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    
    safe_name = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIR, safe_name)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    index_name = safe_name.replace(".pdf", "")
    ingest_pdf(file_path, index_name)
    
    return {"message": "PDF uploaded and processed!", "index_name": index_name}

@app.post("/ask")
async def ask(request: QuestionRequest):
    answer = ask_question(request.index_name, request.question)
    return {"answer": answer}

@app.post("/generate-quiz")
async def quiz(request: QuizRequest):
    questions = generate_quiz(
        request.index_name,
        request.topic,
        request.num_questions
    )
    return {"questions": questions}