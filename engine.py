import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

# 1. HYBRID LLM SWITCH
def get_llm(is_cloud, st_secrets):
    if is_cloud:
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    return ChatOllama(model="gemma2:2b", temperature=0)

# 2. SMART-SCAN KNOWLEDGE BASE
def load_multi_knowledge_base(selected_list, root_folder="Regulations"):
    """Scans all subfolders in /Regulations to find PDFs."""
    all_chunks = []
    
    if not os.path.exists(root_folder):
        return None

    # Walk through the directory to find every PDF
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                full_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    
                    for d in docs:
                        d.metadata["source_file"] = file
                    
                    # Optimization: Larger overlap ensures legal definitions stay intact
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1200, 
                        chunk_overlap=250
                    )
                    all_chunks.extend(splitter.split_documents(docs))
                except Exception:
                    continue
    
    if not all_chunks:
        return None
    
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# 3. PDF GENERATOR
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "READY-AUDIT: REGULATORY COMPLIANCE REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())
