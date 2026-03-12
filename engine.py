import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

# 1. THE BRAIN SWITCH (Handles Chromebook vs. Cloud)
def get_llm(is_cloud, st_secrets):
    if is_cloud:
        # For the website version
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    # For your Chromebook (Local)
    return ChatOllama(model="gemma2:2b", temperature=0)

# 2. THE MULTI-FOLDER KNOWLEDGE BASE
def load_multi_knowledge_base(selected_list, framework_folders):
    """Loads PDFs from multiple selected folders into one database."""
    all_chunks = []
    for framework in selected_list:
        path = framework_folders.get(framework, ".")
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(path, f))
                    docs = loader.load()
                    # Tag each page so the AI knows which law it's reading
                    for d in docs:
                        d.metadata["framework"] = framework
                    
                    # Splitting logic
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    all_chunks.extend(splitter.split_documents(docs))
    
    if not all_chunks:
        return None
        
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# 3. THE PDF EXPORTER
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL MULTI-FRAMEWORK AUDIT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())
