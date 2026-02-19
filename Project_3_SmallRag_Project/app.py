'''
import os 
from dotenv import load_dotenv

from rag.loader import load_and_split_doc
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstore
from rag.retriever import retrieve
from rag.chain import rag_agent
load_dotenv()

API_KEY=os.getenv("GROQ_API_KEY")


documents=load_and_split_doc()

embeddings=get_embeddings()

vectorstore=create_vectorstore(documents,embeddings)

retriever=retrieve(vectorstore)

rag_chain=rag_agent(retriever)

while True:
    query=input("\nAsk a Question: (type 'exit' to quit)")
    response=rag_chain.invoke({"input":query})
    print("\nAnswer: ",response["answer"])'''

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.loader import load_and_split_doc
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vectorstore
from rag.retriever import retrieve
from rag.chain import rag_agent

# Load environment
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Initialize app
app = FastAPI(title="RAG Assistant")

# Load RAG once at startup
documents = load_and_split_doc()
embedding = get_embeddings()
vectorstore = create_vectorstore(documents, embedding)
retriever = retrieve(vectorstore)
rag_chain = rag_agent(retriever, API_KEY)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# Request model
class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(q: Question):
    response = rag_chain.invoke({"input": q.question})
    return {"answer": response["answer"]}
