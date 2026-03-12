import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from fpdf import FPDF

def get_llm(is_cloud, st_secrets):
    if is_cloud:
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_multi_knowledge_base(selected_list, root_folder="Regulations"):
    """
    Scans the ENTIRE Regulations folder and all subfolders 
    to find any PDF files that match your selected frameworks.
    """
    all_chunks = []
    
    if not os.path.exists(root_folder):
        return None

    # This 'walks' through every single folder in your GitHub Regulations directory
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                # Check if the filename or the folder name matches what the user selected
                # This makes the search very 'forgiving'
                full_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(full_path)
                    docs = loader.load()
                    
                    # Label the context so the AI knows which file it's reading
                    for d in docs:
                        d.metadata["source_file"] = file
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    all_chunks.extend(splitter.split_documents(docs))
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
    
    if not all_chunks:
        return None
    
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "READY-AUDIT COMPLIANCE REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())
