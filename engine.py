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

def find_and_scrape_company(company_name, tavily_key=None):
    if not tavily_key: return "Manual verification required: No search API key found."
    try:
        os.environ["TAVILY_API_KEY"] = tavily_key
        search = TavilySearchResults(k=5) # Increased k for deeper scouting
        # Target keywords for the "Beast" calculator: job displacement, automation, workforce reduction
        query = f"{company_name} AI automation workforce impact hiring algorithms 2026"
        return str(search.run(query))
    except Exception as e:
        return f"Scouting failed: {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        # 1. Operational "Robot Tax" estimate (0.05% of token-based savings)
        token_tax = (token_usage / 1000) * 0.0005
        
        # 2. COLORADO SB 24-205 PENALTY: $20,000 per violation (per impacted person)
        # This is the "Beast" logic. If 50 roles are replaced without an impact assessment, that's 50 violations.
        statutory_penalty = replaced_staff * 20000 
        
        # 3. CLASS ACTION MULTIPLIER (Est. legal defense costs)
        legal_buffer = statutory_penalty * 0.25 
        
        return {
            "statutory": statutory_penalty,
            "total": round(token_tax + statutory_penalty + legal_buffer, 2)
        }

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
