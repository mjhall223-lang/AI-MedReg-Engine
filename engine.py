import os
import re
import streamlit as st
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# 2026 Standard: Using DDGS context manager for privacy-first search
from duckduckgo_search import DDGS

def get_llm(is_cloud, st_secrets):
    """Initializes LLM based on environment."""
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def extract_headcount(text, llm):
    """Dynamic Sifter: Finds the 'Magic Number' (headcount) in raw news text."""
    prompt = f"Identify the specific number of people affected (layoffs or trial participants) in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Returns found number or default to 10 if extraction fails
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_and_scrape_live_news(company_name):
    """2026 Web Sifter: Scrapes DuckDuckGo for real-time liability triggers."""
    try:
        with DDGS() as ddgs:
            # Sifts for specific 2026 keywords for Block (layoffs) or Synchron (trials)
            query = f"March 2026 {company_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search sifter offline: {e}"

def load_selected_docs(active_files, root_folder="Regulations"):
    """Loads and vectors local regulatory PDFs."""
    all_chunks = []
    if not active_files: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for file in active_files:
        f_path = os.path.join(root_folder, file)
        if os.path.exists(f_path):
            try:
                loader = PyPDFLoader(f_path)
                all_chunks.extend(splitter.split_documents(loader.load()))
            except: continue
    if not all_chunks: return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

class EconomicImpact:
    """Calculates statutory debt based on Colorado SB 24-205."""
    @staticmethod
    def calculate_liability(headcount=0):
        # 2026 Statutory Rate: $20,000 per violation
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    """Generates a bytes-compatible PDF for Streamlit download."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    # Sanitize text for latin-1 encoding
    clean_text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    
    # 2026 Streamlit Fix: Return output as raw bytes to prevent 'Unsupported Error'
    return bytes(pdf.output())
