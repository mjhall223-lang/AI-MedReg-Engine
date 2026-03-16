import io
import PyPDF2
from pathlib import Path
from langchain_groq import ChatGroq

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """Recursive Hunter: Finds all PDFs in your nested Regulations folders."""
    path_root = Path(base_dir)
    # rglob finds files in all subdirectories automatically
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
