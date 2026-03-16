import io
import PyPDF2
from pathlib import Path
from langchain_groq import ChatGroq

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """Recursive Hunter: Walks through nested Regulations/Regulations folders."""
    path_root = Path(base_dir)
    # .rglob('*') handles any amount of folder nesting automatically
    return sorted([str(f.relative_to(path_root.parent)) for f in path_root.rglob('*.pdf')])

def extract_pdf_text(uploaded_file):
    """SAFE EXTRACTOR: Prevents UnicodeDecodeError by reading binary streams."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        return text
    except Exception as e:
        return f"Extraction Error: {e}"

def perform_gap_analysis(file_content, selected_laws, org_name, llm):
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS for {org_name}.
    Auditing against: {', '.join(selected_laws)}
    
    Technical File: {file_content[:4000]}
    
    TASKS:
    1. Identify 'The Hole' in this architecture.
    2. List remediation steps to fill it before June 30, 2026.
    """
    return llm.invoke(gap_prompt).content
