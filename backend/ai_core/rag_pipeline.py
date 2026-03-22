import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ai_core.embeddings import load_vectorstore

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")

prompt_template = PromptTemplate.from_template("""
You are StudyMind AI, a helpful study assistant for college students.
Use the following context from the student's notes to answer the question.
If the answer is not in the context, say "I couldn't find that in your notes."

Context:
{context}

Question: {question}

Give a clear, concise answer as if explaining to a student:""")

def ask_question(index_name: str, question: str):
    vectorstore = load_vectorstore(index_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)