import os
import re
import streamlit as st
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
    """SIFTER: Finds the headcount (e.g., 4000 for Block) in raw news text."""
    prompt = f"Identify the specific number of people affected (layoffs or trial participants) in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # 4000 for Block, 50 for Synchron. This handles those specifically.
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_and_scrape_live_news(company_name):
    """2026 Web Sifter: Scrapes for specific March 2026 triggers."""
    try:
        with DDGS() as ddgs:
            query = f"March 2026 {company_name} AI automation layoffs clinical trial participants"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except Exception:
        return "Search offline."

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # 2026 Statutory Rate: $20,000 per violation (CO SB 24-205)
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    """Generates a bytes-compatible PDF to fix Streamlit's 'unsupported_error'."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    clean_text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    return bytes(pdf.output())
