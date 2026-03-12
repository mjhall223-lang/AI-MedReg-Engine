import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

# 1. THE HYBRID SWITCH (Detects Cloud vs. Chromebook)
def get_llm(is_cloud, st_secrets):
    if is_cloud:
        # High-speed Cloud Mode for the website
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    # Secure Local Mode for your Chromebook
    return ChatOllama(model="gemma2:2b", temperature=0)

# 2. THE MULTI-FOLDER KNOWLEDGE BASE (The "Librarian")
def load_multi_knowledge_base(selected_list, framework_folders):
    """Loops through ALL selected frameworks and grabs their PDFs."""
    all_chunks = []
    
    for framework in selected_list:
        # Get the path from our dictionary
        path = framework_folders.get(framework, ".")
        
        # Check if the folder exists
        if os.path.exists(path):
            # Find all PDFs in that specific folder
            pdf_files = [f for f in os.listdir(path) if f.endswith(".pdf")]
            
            for f in pdf_files:
                try:
                    full_path = os.path.join(path, f)
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    
                    # Tag each page so the AI knows which framework it belongs to
                    for d in docs:
                        d.metadata["framework"] = framework
                    
                    # Break long laws into searchable pieces
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    all_chunks.extend(splitter.split_documents(docs))
                except Exception:
                    continue # Skip any corrupted PDFs
    
    if not all_chunks:
        return None
    
    # Create the searchable database (stays local/CPU)
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# 3. THE PDF EXPORTER
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL READY-AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    # Clean up text for PDF encoding
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())
