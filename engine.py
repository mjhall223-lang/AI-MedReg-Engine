import io
import pypdf  # Updated for 2026 stability
from pathlib import Path
from langchain_groq import ChatGroq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    path_root = Path(base_dir)
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    """SAFE BINARY EXTRACTOR: Uses pypdf to prevent Unicode errors."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        return text
    except Exception as e:
        return f"Extraction Error: {e}"

def generate_pdf_report(results, org_name, hole_type, selected_laws):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"REMEDIATION AUDIT: {org_name}", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Financial Exposure Box
    story.append(Paragraph("FINANCIAL EXPOSURE SUMMARY", styles['Heading2']))
    penalty = "$20,000 per violation" if "Human" in hole_type else "$10,000 per violation"
    story.append(Paragraph(f"Statutory Risk: {penalty}", styles['Normal']))
    story.append(Paragraph(f"Active Deadline: June 30, 2026", styles['Normal']))
    story.append(Paragraph(f"Audited Against: {', '.join(selected_laws)}", styles['Italic']))
    story.append(Spacer(1, 18))

    story.append(Paragraph("DETAILED GAP ANALYSIS", styles['Heading3']))
    for line in results.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer
