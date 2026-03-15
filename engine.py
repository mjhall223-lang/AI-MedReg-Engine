import os
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
    all_chunks = []
    if not active_files: return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    for root, _, files in os.walk(root_folder):
        for file in files:
            if file in active_files:
                try:
                    loader = PyPDFLoader(os.path.join(root, file))
                    docs = loader.load()
                    all_chunks.extend(splitter.split_documents(docs))
                except: continue
    if not all_chunks: return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

def find_and_scrape_company(company_name, tavily_key=None):
    try:
        if not tavily_key: return "Technical Note: No Tavily Key provided. Manual search required."
        os.environ["TAVILY_API_KEY"] = tavily_key
        search = TavilySearchResults(k=3)
        query = f"{company_name} AI ethics policy medical device compliance disclosures"
        results = search.run(query)
        return str(results)
    except Exception as e:
        return f"Technical Note: Search failed - {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        token_tax = (token_usage / 1000) * 0.0005
        payroll_tax = (replaced_staff * 60000) * 0.15
        return {"total": round(token_tax + payroll_tax, 2)}

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
