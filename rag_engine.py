# rag_engine.py
import os
import time
import requests
from dotenv import load_dotenv
from langdetect import detect
from googletrans import Translator

from Bio import Entrez
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_xml_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from google.api_core.exceptions import ServiceUnavailable

# ─────────────────── ENV SETUP ─────────────────── #
load_dotenv()
USER_AGENT = os.getenv("USER_AGENT", "medbot/1.0 (contact:engineervishal007@gmail.com)")
Entrez.email = os.getenv("ENTREZ_EMAIL")
Entrez.api_key = os.getenv("ENTREZ_API_KEY")
Entrez.tool = USER_AGENT
os.environ["USER_AGENT"] = USER_AGENT
translator = Translator()

# ─────────────────── HELPERS ─────────────────── #

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def translate(text: str, src: str, dest: str) -> str:
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        print(f"[Translation Error]: {e}")
        return text

def fetch_pubmed(query: str, n=3) -> str:
    try:
        h = Entrez.esearch(db="pubmed", term=query, retmax=str(n))
        ids = Entrez.read(h).get("IdList", [])
        if not ids:
            return ""
        return Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text").read()
    except Exception as e:
        print(f"[PubMed Error]: {e}")
        return ""

def fetch_clinical_trials(query: str) -> str:
    try:
        search_url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize=3"
        resp = requests.get(search_url, headers={"User-Agent": USER_AGENT}, timeout=20)
        if resp.status_code != 200:
            print(f"[ERROR] ClinicalTrials.gov API v2 responded with {resp.status_code}")
            return ""
        results = resp.json().get("studies", [])
        summaries = [
            f"Title: {s.get('briefTitle', '')}\nSummary: {s.get('briefSummary', '')}"
            for s in results
        ]
        return "\n\n".join(summaries)
    except Exception as e:
        print(f"[ClinicalTrials Error]: {e}")
        return ""

# ─────────────────── DATA LOADING ─────────────────── #

print("[INFO] Loading WHO general factsheets (for vector base)...")
who_docs = []
try:
    who_docs.extend(WebBaseLoader("https://www.who.int/news-room/fact-sheets/").load())
except Exception as e:
    print(f"[WHO LOADER] Error: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(who_docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if os.path.exists("faiss_store/index.faiss"):
    print("[INFO] Loading FAISS index from local...")
    vector_db = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
else:
    print("[INFO] Creating and saving FAISS index...")
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_store")

# ─────────────────── TOOLS & AGENT ─────────────────── #

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=1000))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=5, doc_content_chars_max=1000))
retriever_tool = create_retriever_tool(
    retriever=vector_db.as_retriever(),
    name="med-info-search",
    description="Search WHO, PubMed, and ClinicalTrials data"
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

prompt_template = """
You are a medical assistant powered by Gemini. Answer clearly and concisely.

Languages supported: English, Hindi, Gujarati, Marathi, Punjabi, Bengali, French, Spanish, German, Tamil, Telugu, Malayalam, Kannada.

Use the following tools if needed:
{tools}

Wrap tool usage in <tool> tags. Wrap final response in <final_answer> ... </final_answer>.

If user asks "in detail", try to respond in ~550 words. Otherwise ~350.

If none of the relevant information is fetched via tools, use "Gemini" capabilities

Previous chat:
{chat_history}

User query:
{input}

{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(prompt_template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output", human_prefix="Human", ai_prefix="ai")

agent = create_xml_agent(llm, [retriever_tool, wiki_tool, arxiv_tool], prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[retriever_tool, wiki_tool, arxiv_tool],
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    output_key="output"
)

# ─────────────────── MAIN QUERY FUNCTION ─────────────────── #

def run_query(user_input: str) -> dict:
    print(f"\n[QUERY] {user_input}")
    lang_code = detect_language(user_input)
    print(f"[LANGUAGE] Detected: {lang_code}")

    translated_input = translate(user_input, src=lang_code, dest="en")
    print(f"[TRANSLATED] Query: {translated_input}")

    sources_used = []
    trials_text = fetch_clinical_trials(translated_input)
    if trials_text:
        vector_db.add_documents([Document(page_content=trials_text, metadata={"source": "ClinicalTrials.gov"})])
        sources_used.append("ClinicalTrials.gov")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = agent_executor.invoke({"input": translated_input})
            final_output = result["output"]
            final_translated = translate(final_output, src="en", dest=lang_code)

            return {
                "query": user_input,
                "translated_query": translated_input,
                "response": final_translated,
                "sources": list(set(sources_used))
            }
        except ServiceUnavailable:
            print("[RETRY] Gemini API overloaded. Retrying...")
            time.sleep(5)
        except Exception as e:
            print(f"[ERROR] {e}")
            break

    return {
        "query": user_input,
        "response": "Unable to get a response due to system overload or error.",
        "sources": []
    }
