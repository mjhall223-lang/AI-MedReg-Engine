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
    """Dynamic Sifter: Finds the 'Beast Number' (headcount) in raw news text."""
    prompt = f"Identify the specific number of people affected (layoffs/participants) in this text: {text[:2500]}. Output ONLY the digits."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    # Default to 10 if nothing found, but Cap at 1M to avoid hallucinations
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

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # 2026 Statutory Rate: $20,000 per violation (CO SB 24-205)
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
    
    # CRITICAL 2026 FIX: Must return as raw bytes to prevent 'Unsupported Error'
    return bytes(pdf.output())
