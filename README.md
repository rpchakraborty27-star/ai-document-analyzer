# AI Document Analyzer

A full-stack **Retrieval-Augmented Generation (RAG)** web application for uploading PDF documents, generating structured executive summaries, and asking analytical questions against document content.

Built with **FastAPI**, **LangChain**, **FAISS**, **HuggingFace Sentence Transformers**, **Google Gemini**, and a responsive **Tailwind CSS + Marked.js** frontend.

---

## Key Features

- **PDF Ingestion & Executive Summarisation:** Extracts multi-page PDF text via `PyPDFLoader`, splits content into overlapping semantic chunks, and generates formatted Markdown summaries.
- **Context-Aware Document Q&A:** Indexes document embeddings locally in a FAISS vector store and retrieves the top relevant passages (`k=6`) for grounded, analytical answers—including career and role-suitability inferences for CVs.
- **Local Zero-Cost Embeddings:** Runs `all-MiniLM-L6-v2` locally via `HuggingFaceEmbeddings`, avoiding external embedding API limits and latency.
- **Clean Markdown Rendering:** Translates raw LLM Markdown responses into styled headings, dividers, and lists using **Marked.js** and the **Tailwind CSS Typography** plugin.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | Python 3.13, FastAPI, Uvicorn |
| **AI & Orchestration** | LangChain, Google Gemini (`langchain-google-genai`) |
| **Embeddings & Vector DB** | HuggingFace (`all-MiniLM-L6-v2`), FAISS (in-memory cache) |
| **Document Parsing** | `PyPDFLoader`, `RecursiveCharacterTextSplitter` |
| **Frontend** | HTML5, Vanilla JavaScript, Tailwind CSS (Typography Plugin), Marked.js |

---

## Architecture & Engineering Highlights

1. **Global Model Preloading:** The HuggingFace PyTorch embedding model is initialised once at server startup rather than inside the request handler, reducing per-document processing time from minutes to seconds.
2. **ThreadPool Concurrency (`def` vs `async def`):** FastAPI endpoints are declared synchronously so CPU-bound PyTorch embedding operations and blocking file I/O run in background worker threads instead of stalling the main ASGI event loop.
3. **Persistent In-Memory Cache:** Uvicorn runs with `reload=False` so temporary PDF uploads written to `temp_docs/` do not trigger server restarts and wipe the active FAISS `vector_db_cache`.

---

## Project Structure

```text
ai-document-analyzer/
├── main.py              # FastAPI backend, RAG pipeline, and endpoints
├── index.html           # Frontend dashboard (Tailwind CSS + Marked.js)
├── requirements.txt     # Pinned Python dependencies
├── .env.example         # Environment variable template
├── .gitignore           # Excludes .env, venv/, and temp_docs/
└── README.md            # Project documentation
```

---

## Getting Started (Local Setup)

### Prerequisites
- Python 3.10+
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<rpchakraborty27-star>/ai-document-analyzer.git
cd ai-document-analyzer
```

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --default-timeout=1000 -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root (based on `.env.example`) and add your Gemini API key:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```

### 5. Start the Backend Server
```bash
python main.py
```
Wait until the terminal displays `INFO: Application startup complete.` The server will run on `http://127.0.0.1:8001`.

### 6. Launch the Frontend
Open `index.html` directly in Chrome or Edge (right-click `index.html` in VS Code → **Copy Path** → paste into your browser address bar).

---

## API Endpoints

| Method | Endpoint | Input | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | None | Health check confirming the server is active |
| `POST` | `/summarize/` | `file` (`multipart/form-data` PDF) | Parses PDF, indexes chunks in FAISS, and returns an executive summary |
| `POST` | `/ask/` | `filename`, `question` (`form-data`) | Runs semantic search (`k=6`) against the indexed PDF and generates an analytical answer |

---

## Security Note
Never commit your `.env` file or personal uploaded PDFs. Both `.env` and `temp_docs/` are excluded by default in `.gitignore`.