import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI (Bypassed for offline testing)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import uvicorn

load_dotenv()

app = FastAPI(title="Enterprise Document AI", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_docs", exist_ok=True)
vector_db_cache = {}

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@app.get("/")
def health_check():
    return {"status": "Active", "message": "Server running securely."}


@app.post("/summarize/")
def summarize_document(file: UploadFile = File(...)):
    try:
        print(f"\n--- Starting process for {file.filename} ---")
        
        file_path = f"temp_docs/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        print(f"Step 1: PDF loaded ({len(docs)} pages).")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        vector_db = FAISS.from_documents(splits, embeddings)
        vector_db_cache[file.filename] = vector_db
        print("Step 2: Local Vector DB built instantly.")

        print("Step 3: Bypassing Gemini API, generating dummy markdown...")
        
        # DUMMY SUMMARY DATA
        dummy_summary = """
### 🚀 Simulated Executive Summary
**Candidate Profile Overview**
This is a dummy summary generated to test the pipeline while the cloud AI is offline. Your backend successfully ingested, chunked, and indexed the PDF!

**Core Pipeline Status:**
* **PDF Parsing**: Success
* **Vector Embeddings**: Success (Local CPU)
* **Frontend Rendering**: Success (Marked.js & Tailwind)
"""
        
        print("Step 4: Pipeline complete.")

        return {
            "filename": file.filename,
            "status": "Summary Generated Successfully",
            "total_pages": len(docs),
            "chunks_indexed": len(splits),
            "summary": dummy_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask/")
def ask_question(filename: str = Form(...), question: str = Form(...)):
    if filename not in vector_db_cache:
        raise HTTPException(
            status_code=404, 
            detail=f"Document '{filename}' not found in memory. Please upload and summarize it first."
        )

    try:
        vector_db = vector_db_cache[filename]
        
        # We still run the real semantic search to prove FAISS is working!
        relevant_docs = vector_db.similarity_search(question, k=6)
        
        # DUMMY Q&A DATA
        dummy_answer = f"""
### 🎯 Simulated HR Analysis
*This is a simulated response to test the Q&A workflow.*

**Your Question:** {question}

**Recommended Job Roles (Simulated Logic):**
1. **AI/ML Engineer**: Strong backend skills (FastAPI, Python) and AI integration experience.
2. **Computational Design Engineer**: Matches the mechanical engineering and CFD background.
3. **Full-Stack Developer**: Leverages the Angular and UI experience.

*System check: The FAISS vector database successfully scanned your CV and retrieved **{len(relevant_docs)}** highly relevant text chunks matching your question!*
"""
        
        return {
            "filename": filename,
            "question": question,
            "answer": dummy_answer,
            "chunks_referenced": len(relevant_docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running Q&A: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=False)