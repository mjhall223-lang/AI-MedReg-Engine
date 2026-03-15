import os
import streamlit as st
import re
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from ddgs import DDGS # Updated to the 2026 library

def get_llm(is_cloud, st_secrets):
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def extract_headcount(text, llm):
    """Dynamic Sifter: Finds the 'Magic Number' in the news."""
    prompt = f"Identify the specific number of people affected (layoffs/participants) in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_and_scrape_live_news(company_name):
    """2026 Search: Sifts headlines for Synchron or Block."""
    try:
        results = DDGS().text(f"March 2026 {company_name} AI layoffs headcount clinical trial", max_results=5)
        return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search offline: {e}"

def load_selected_docs(active_files, root_folder="Regulations"):
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
    @staticmethod
    def calculate_liability(headcount=0):
        statutory = headcount * 20000 # CO SB 24-205 standard
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=clean)
    return pdf.output()
