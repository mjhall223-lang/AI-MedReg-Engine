import os
import datetime
import streamlit as st
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def get_llm(is_cloud, st_secrets):
    """Initializes the LLM based on environment."""
    if is_cloud:
        from langchain_groq import ChatGroq
        # Llama 3.3 70B is the 2026 workhorse for complex policy analysis
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_multi_knowledge_base(selected_frameworks, selected_fed_docs=[], root_folder="Regulations"):
    """Crawl regulations and create a FAISS vector store."""
    all_chunks = []
    if not os.path.exists(root_folder): 
        return None
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                path_lower = root.lower()
                is_framework = any(f.split()[0].lower() in path_lower for f in selected_frameworks)
                is_toggled_doc = (file in selected_fed_docs)
                
                if is_framework or is_toggled_doc:
                    try:
                        loader = PyPDFLoader(os.path.join(root, file))
                        docs = loader.load()
                        for d in docs: 
                            d.metadata["source_file"] = file
                        all_chunks.extend(splitter.split_documents(docs))
                    except:
                        continue
                        
    if not all_chunks:
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        """Simulated 2026 'Robot Tax' and 'Divergence' metrics."""
        token_tax = (token_usage / 1000) * 0.0005
        payroll_tax = (replaced_staff * 60000) * 0.15
        return {
            "token_tax": round(token_tax, 2), 
            "payroll_tax": round(payroll_tax, 2), 
            "total": round(token_tax + payroll_tax, 2)
        }

def create_pdf(text):
    """Generates the certified audit report PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "READY-AUDIT: CERTIFIED REGULATORY REPORT", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=11)
    # Sanitize text for FPDF latin-1
    clean_text = text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'")
    pdf.multi_cell(0, 10, txt=clean_text.encode('latin-1', 'replace').decode('latin-1'))
    
    pdf.ln(20)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "OFFICIAL CERTIFICATION:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", ln=True)
    pdf.cell(0, 10, f"Lead Specialist: Myia Hall", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "X________________________________________", ln=True)
    pdf.cell(0, 10, "Signature of Senior Regulatory Architect", ln=True)
    
    return bytes(pdf.output())
