# 🧠 LLM API – Multilingual Medical Assistant with RAG

This FastAPI-based application provides a **multilingual conversational assistant for medical queries**, powered by **Gemini Pro (via LangChain)** and a **custom Retrieval-Augmented Generation (RAG)** pipeline using sources like:

- ✅ WHO Fact Sheets  
- ✅ ClinicalTrials.gov  
- ✅ PubMed abstracts  
- ✅ Wikipedia  
- ✅ ArXiv (optional academic support)

It supports **Indian and international languages** like Hindi, Gujarati, Marathi, Tamil, Telugu, French, German, Spanish, and more.

---

## 🚀 Features

- 🌍 Multilingual language detection & translation
- 🧠 Gemini Pro for final response generation
- 🧾 Integrated data from WHO, PubMed, and ClinicalTrials
- 📚 FAISS-based vector search
- 🧰 XML Agent with LangChain tool orchestration
- 🧠 Handles follow-ups with conversational memory

---


## 📁 Project Structure

```bash
llm_api/
├── main.py # FastAPI entrypoint
├── rag_engine.py # Core RAG + Gemini logic
├── faiss_store/ # FAISS vector index (auto-generated or preloaded)
├── requirements.txt # All Python dependencies
└── .env # API keys (not committed)
```
---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/22AIML021VishalMehta/LLM_RAG_Component
cd LLM_RAG_Component
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root of `LLM_RAG_Component/`:

```bash
GOOGLE_API_KEY=your_gemini_api_key
ENTREZ_API_KEY=your_pubmed_key
ENTREZ_EMAIL=your_email@example.com
USER_AGENT=medbot/1.0 (contact:you@example.com)
```

## ▶️ Run the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
## 📡 API Endpoint
POST `/query`

### Request JSON:

```bash
{
  "query": "What are the latest treatments for type 2 diabetes?"
}
```

### Response:
```bash
{
  "query": "What are the latest treatments for type 2 diabetes?",
  "response": "Based on recent medical data...",
  "sources": ["ClinicalTrials.gov", "PubMed"]
}
```

# 📌 Notes
- If `faiss_store/` is missing, it will auto-generate the vector index from WHO data.

- Gemini Pro is used via LangChain’s `ChatGoogleGenerativeAI`.

- ClinicalTrials.gov and PubMed content is fetched in real time.

- All queries are translated to English and then back to the user’s language.