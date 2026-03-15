import os
import datetime
import requests
import streamlit as st
from bs4 import BeautifulSoup
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

def get_llm(is_cloud, st_secrets):
    """Initializes the LLM connection for the audit."""
    if is_cloud:
        from langchain_groq import ChatGroq
        return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_selected_docs(active_files, root_folder="Regulations"):
    """Crawls and indexes only the PDFs toggled 'ON' in the sidebar."""
    all_chunks = []
    if not active_files: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file in active_files:
                try:
                    loader = PyPDFLoader(os.path.join(root, file))
                    docs = loader.load()
                    for d in docs: d.metadata["source_file"] = file
                    all_chunks.extend(splitter.split_documents(docs))
                except: continue
    if not all_chunks: return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def find_and_scrape_company(company_name):
    """Autonomous Scout: Searches the web and scrapes public policy data."""
    try:
        search = TavilySearchResults(k=2)
        query = f"{company_name} AI transparency policy terms of service"
        results = search.run(query)
        scraped_content = ""
        for res in results:
            try:
                url = res['url']
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                scraped_content += "\n".join([p.get_text() for p in soup.find_all('p')])
            except: continue
        return scraped_content[:15000]
    except Exception as e:
        return f"Search Error: {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        token_tax = (token_usage / 1000) * 0.0005
        payroll_tax = (replaced_staff * 60000) * 0.15
        return {"total": round(token_tax + payroll_tax, 2)}

def create_pdf(text, title="CERTIFIED AUDIT REPORT"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, title, ln=True, align='C')
    pdf.set_font("Arial", size=11)
    clean_text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())
    
