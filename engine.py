import os
import streamlit as st
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def get_llm(is_cloud, st_secrets):
    """Initializes the LLM based on environment."""
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_selected_docs(active_files, root_folder="Regulations"):
    """Builds the searchable knowledge base from selected PDFs."""
    all_chunks = []
    if not active_files: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file in active_files:
                try:
                    loader = PyPDFLoader(os.path.join(root, file))
                    all_chunks.extend(splitter.split_documents(loader.load()))
                except: continue
    if not all_chunks: return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def find_and_scrape_company(company_name, tavily_key=None):
    """Deep-scrapes for workforce impact and 2026 AI product data."""
    if not tavily_key: return "Technical Note: No search API key provided."
    try:
        os.environ["TAVILY_API_KEY"] = tavily_key
        search = TavilySearchResults(k=5)
        # Targeted 2026 query for the 'Beast' calculator
        query = f"{company_name} AI automation layoffs 2026 neural data privacy"
        return str(search.run(query))
    except Exception as e:
        return f"Scouting failed: {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        """The 'Beast' Calculator: $20,000 per violation (per person)."""
        # 1. Statutory Fine (Colorado SB 24-205)
        statutory_penalty = replaced_staff * 20000 
        # 2. Operational token-based tax estimate
        token_tax = (token_usage / 1000) * 0.0005
        # 3. Legal/Defense Buffer (25%)
        legal_buffer = statutory_penalty * 0.25 
        
        return {
            "statutory": statutory_penalty,
            "total": round(token_tax + statutory_penalty + legal_buffer, 2)
        }

def create_pdf(text, title="READY-AUDIT CERTIFIED REPORT"):
    """Generates PDF bytes with standard Latin-1 encoding for stability."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=11)
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
