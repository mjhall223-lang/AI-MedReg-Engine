import io, requests
from pathlib import Path
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def perform_gap_analysis(content, laws, org, llm):
    """STRICT 2026 AUDITOR: Identifies $20k violations and missing artifacts."""
    system_prompt = f"""
    You are a Lead Federal Policy & Compliance Strategist. 
    Audit Date: March 16, 2026. Cliff: June 30, 2026.
    
    IDENTITY: You are auditing {org} against {laws}. 
    
    STRICT INSTRUCTIONS:
    1. Identify if the tech performs 'Intention-to-Action' or 'Neural Decoding'.
    2. Under Colorado SB 24-205 (as amended by SB 25B-004), these are 'Consequential Decisions'.
    3. CHECK FOR: A 'Human Appeal Path' and an 'Adverse Decision Notice'.
    4. FINANCIAL RISK: If missing, cite the $20,000 statutory penalty per violation.
    5. NEURAL DATA: Flag 'deidentification' as a failed 2026 privacy shield.
    
    OUTPUT: Must include '🛑 CRITICAL VIOLATIONS' and '📁 REQUIRED REMEDIATION ARTIFACTS'.
    """

    audit_query = f"Audit the following for {org}:\n\n{content[:5000]}"
    
    response = llm.invoke([
        ("system", system_prompt),
        ("human", audit_query)
    ])
    return response.content

# Keep existing helper functions (list_all_laws, smart_web_sifter, extract_pdf_text, generate_pdf_report)
