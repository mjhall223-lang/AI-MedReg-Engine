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
    """SIFTER: Finds the 2026 'Beast Number' (e.g. 4000 for Block)."""
    prompt = f"Analyze March 2026 news: {text[:2500]}. Find the specific number of employees laid off or trial participants. Output ONLY the integer."
    response = llm.invoke(prompt).content
    number = re.sub(r"\D", "", response)
    return int(number) if (number and 0 < len(number) < 8) else 10

def find_and_scrape_live_news(company_name):
    try:
        with DDGS() as ddgs:
            # Sifts for the actual Dorsey restructuring or Synchron COMMAND trial data
            query = f"March 2026 {company_name} AI automation layoffs BCI enrollment news"
            results = list(ddgs.text(query, max_results=5))
            return "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: return "Search failed."

class EconomicImpact:
    @staticmethod
    def calculate_liability(headcount=0):
        # Statutory: $20,000 per violation (CO SB 24-205)
        statutory = headcount * 20000 
        return {"statutory": statutory, "total": round(statutory * 1.25, 2)}

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    # Cleaning for FPDF encoding
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
