import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

# 1. THE HYBRID SWITCH
def get_llm(is_cloud, st_secrets):
    if is_cloud:
        # High-performance Cloud Mode for the website
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    # Secure Local Mode for your Chromebook
    return ChatOllama(model="gemma2:2b", temperature=0)

# 2. THE MULTI-FRAMEWORK KNOWLEDGE BASE
def load_multi_knowledge_base(selected_list, framework_folders):
    all_chunks = []
    for framework in selected_list:
        path = framework_folders.get(framework, ".")
        # We check both the folder and if it actually contains PDFs
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(".pdf")]
            for f in files:
                try:
                    loader = PyPDFLoader(os.path.join(path, f))
                    docs = loader.load()
                    for d in docs:
                        d.metadata["framework"] = framework
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    all_chunks.extend(splitter.split_documents(docs))
                except Exception:
                    continue # Skip broken PDFs
    
    if not all_chunks:
        return None
    
    # Using MiniLM for fast, local-style embeddings
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# 3. PDF EXPORT LOGIC
def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL READY-AUDIT REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())
