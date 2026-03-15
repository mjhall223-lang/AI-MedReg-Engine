import os
import streamlit as st
import re
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient

def get_llm(is_cloud, st_secrets):
    if is_cloud and st_secrets.get("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def extract_headcount(text, llm):
    """Uses LLM to sift through news and find the headcount number."""
    prompt = f"""
    Analyze the following news text and extract the specific number of people affected 
    (e.g., number of employees laid off, clinical trial participants, or users).
    
    NEWS: {text[:2000]}
    
    Rules:
    - Output ONLY the integer.
    - If a range is given (e.g. 30-50), output the higher number.
    - If no number is found, output '10'.
    """
    response = llm.invoke(prompt).content.strip()
    # Clean output to ensure only digits
    number = re.sub(r"\D", "", response)
    return int(number) if number else 10

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

def find_and_scrape_live_news(company_name, tavily_key=None):
    if not tavily_key: return "Tavily API key missing."
    client = TavilyClient(api_key=tavily_key)
    query = f"March 2026 {company_name} AI automation layoffs BCI enrollment"
    results = client.search(query=query, search_depth="advanced", topic="news", max_results=5)
    news_text = ""
    for res in results.get('results', []):
        news_text += f"Headline: {res['title']}\nContent: {res['content']}\n\n"
    return news_text

class EconomicImpact:
    @staticmethod
    def calculate_liability(replaced_staff=0):
        statutory = replaced_staff * 20000 
        buffer = statutory * 0.25 
        return {"statutory": statutory, "total": round(statutory + buffer, 2)}

def create_pdf(text, title="READY-AUDIT CERTIFIED REPORT"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", size=11)
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
