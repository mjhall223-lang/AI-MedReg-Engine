import io
import pypdf 
from pathlib import Path
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """Recursively finds all PDFs in nested folders like Regulations/Regulations/Federal."""
    path_root = Path(base_dir)
    # .rglob('*') handles your nested folder structure automatically
    laws = sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])
    return laws

def generate_pdf_report(results, org_name, hole_type, selected_laws):
    """Professional PDF generator for your remediation reports."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']), Spacer(1, 12)]
    
    # Financial Exposure Summary
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
