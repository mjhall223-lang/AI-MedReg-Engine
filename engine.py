import os
import streamlit as st
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def get_llm(is_cloud, st_secrets):
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_selected_docs(active_files, root_folder="Regulations"):
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
    if not tavily_key: return "Manual verification required: No search API key found."
    try:
        os.environ["TAVILY_API_KEY"] = tavily_key
        search = TavilySearchResults(k=3)
        # 2026 Targeted Search: Finds specific products like 'Chiral' or 'Science Foundry'
        query = f"{company_name} 2026 AI products foundation models compliance"
        return str(search.run(query))
    except Exception as e:
        return f"Search error: {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        # 2026 Colorado AI Act Penalties: Up to $20k per violation
        token_tax = (token_usage / 1000) * 0.0005
        automation_risk = replaced_staff * 20000 
        return {"total": round(token_tax + automation_risk, 2)}

def create_pdf(text, title="READY-AUDIT CERTIFIED REPORT"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", size=11)
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
