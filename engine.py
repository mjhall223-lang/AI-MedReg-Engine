import import io, requests, os
from pathlib import Path
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# --- A-LEVEL INSTRUMENTATION ---
def _enable_instrumentation(st_secrets):
    if "LANGCHAIN_API_KEY" in st_secrets:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = st_secrets["LANGCHAIN_API_KEY"]
        os.environ["LANGCHAIN_PROJECT"] = st_secrets.get("LANGCHAIN_PROJECT", "AI-MedReg-Engine-2026")

# --- REMEDIATION TEMPLATES ---
REMEDIATION_TEMPLATES = {
    "Colorado SB 24-205": ""
