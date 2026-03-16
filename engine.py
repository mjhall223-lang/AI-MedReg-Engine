import io
import pypdf
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """RECURSIVE HUNTER: Walks through nested Regulations/Regulations folders."""
    path_root = Path(base_dir)
    # .rglob('*') handles any amount of folder nesting automatically
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    """SAFE EXTRACTOR: Prevents UnicodeDecodeError by reading binary streams."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        return text
    except Exception as e:
        return f"Extraction Error: {e}"

def web_sifter(org_name):
    with DDGS() as ddgs:
        results = list(ddgs.text(f"{org_name} AI policy 2026", max_results=2))
    if not results: return "No public policy found."
    res = requests.get(results[0]['href'], timeout=10)
    soup = BeautifulSoup(res.text, 'html.parser')
    return f"SOURCE: {results[0]['href']}\n\n" + soup.get_text(separator=' ', strip=True)

def generate_pdf_report(results, org_name, hole_type, selected_laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1, 12)]
    
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph("FINANCIAL EXPOSURE SUMMARY", styles['Heading2']))
    story.append(Paragraph(f"Statutory Risk: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Laws Audited: {', '.join(selected_laws)}", styles['Italic']))
    story.append(Spacer(1, 18))

    for line in results.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))
    doc.build(story)
    buffer.seek(0)
    return buffer
