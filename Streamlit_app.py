import streamlit as st
import os
import tempfile
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Specialist:** MJ Hall")
    audit_framework = st.selectbox(
        "Select Regulatory Framework",
        ["EU AI Act (Medical)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. DYNAMIC MAPPING (Matches your GitHub exactly) ---
framework_folders = {
    "EU AI Act (Medical)": ".",  # Looks in main folder for EU_regulations.pdf & lvdr.pdf
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 / NIST 800-171": "Regulations/Regulations/CMMC" # Matches the typo in your folder name
}
selected_reg_path = framework_folders[audit_framework]

# --- 4. CORE FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    
    if not all_chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

# --- 5. MAIN
