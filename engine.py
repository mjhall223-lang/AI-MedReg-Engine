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

def find_and_scrape_live_news(company_name, tavily_key=None):
    """HUNTER MODE: Scrapes live 2026 news for automation and layoffs."""
    if not tavily_key: return "Technical Note: No search API key provided."
    try:
        os.environ["TAVILY_API_KEY"] = tavily_key
        # 'news' topic ensures we get 2026 headlines, not old 2024 articles
        search = TavilySearchResults(k=5, topic="news", search_depth="advanced")
        query = f"March 2026 {company_name} AI automation layoffs BCI product launch compliance"
        results = search.run(query)
        return str(results)
    except Exception as e:
        return f"Scouting failed: {str(e)}"

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        """THE BEAST: $20,000 per violation (per person) + Legal Buffer."""
        statutory_penalty = replaced_staff * 20000 
        token_tax = (token_usage / 1000) * 0.0005
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
    # Clean characters for PDF stability
    clean = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    pdf.multi_cell(0, 10, txt=clean.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()
