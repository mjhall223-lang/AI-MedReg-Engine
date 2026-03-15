import os
import streamlit as st
import re
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from duckduckgo_search import DDGS

def get_llm(is_cloud, st_secrets):
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def extract_headcount(text, llm):
    """Sifts news for the 'Beast Number' (e.g., Block's 4,000 layoffs)."""
    prompt = f"Extract only the number of people affected (headcount/participants) from this news: {text[:2000]}. Output only digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Default to 10 if sift fails or number is unrealistic
    return int(number) if (number and 0 < len(number) < 7) else 10

def load_selected_docs(active_files, root_folder="Regulations"):
    all_chunks = []
    if not active_files: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for file in active_files:
        file_path = os.path.join(root_folder, file)
        if os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                all_chunks.extend(splitter.split_documents(loader.load()))
            except: continue
    if not all_chunks: return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def find_and_scrape_live_news(company_name):
    """STABLE SEARCH: Sifts March 2026 headlines for liability triggers."""
    try:
        with DDGS() as ddgs:
            query = f"March 2026 {company_name} AI automation layoffs clinical trial enrollment"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(replaced_staff=0):
        # Statutory standard: $20,000 per violation (CO SB 24-205)
        statutory = replaced_staff * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
