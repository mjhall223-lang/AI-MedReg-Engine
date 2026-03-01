import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fpdf import FPDF

# --- 1. CONFIG ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** MJ Hall")
    
    # Updated frameworks based on your new folder structure
    audit_framework = st.selectbox("Framework", [
        "EU AI Act (Medical & IVDR)", 
        "Colorado AI Act", 
        "CMMC 2.0",
        "Medical Bias & Health Equity (Article 10)",
        "FDA PCCP (Clinical Change)"
    ])
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])

# --- 2. GITHUB MAPPING ---
# Matches your exact Regulations/Regulations/Federal structure
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0": "Regulations/Regulations/CMMC",
    "Medical Bias & Health Equity (Article 10)": ".",
    "FDA PCCP (Clinical Change)": "Regulations/Regulations/Federal"
}
selected_reg_path = framework_folders[audit_framework]

# --- 3. FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(loader.load()))
    if not all_chunks:
        return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    # Handle character encoding and return as bytes for Streamlit
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    # Outputting to bytes string
    return bytes(pdf.output())

# --- 4. THE AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")
results_container = st.container()

if st.button("🚀 Run Full Regulatory Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 CROSS-REFERENCING FEDERAL & STATE LAW...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp
