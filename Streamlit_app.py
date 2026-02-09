import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os
import re

# --- PAGE SETUP ---
st.set_page_config(page_title="Bio-AI Compliance Dashboard", page_icon="⚖️", layout="wide")
st.title("⚖️ Bio-AI Compliance & Remediation Engine")
st.subheader("Official Gap Analysis vs. EU AI Act & IVDR (2026)")

# --- 1. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' not found in Secrets.")
    st.stop()

# --- 2. INITIALIZE BRAIN ---
try:
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ CONNECTION ERROR: {e}")
    st.stop()

# --- 3. SIDEBAR: TIERED SERVICE LEVEL ---
with st.sidebar:
    st.markdown("### 🛡️ REGULATORY SHIELD")
    st.info("Version: 1.2.5 (Premium Edition)")
    
    st.markdown("---")
    st.markdown("### 💼 SERVICE LEVEL")
    # This is your business logic toggle
    service_tier = st.radio(
        "Select Analysis Level:", 
        ["Standard Gap Analysis", "Premium Remediation (Consulting)"]
    )
    
    if service_tier == "Premium Remediation (Consulting)":
        st.success("✨ Premium Mode: Strategic Roadmap Enabled")
    
    st.markdown("---")
    st.write(f"**Specialist:** MJ Hall")
    st.write(f"**Affiliation:** Bio-AI Compliance")

# --- 4. LOAD CORE REGULATIONS ---
@st.cache_resource
def load_base_knowledge():
    all_chunks = []
    base_files = ["EU_regulations.pdf", "Ivdr.pdf"]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    for file_name in base_files:
        if os.path.exists(file_name):
            try:
                loader = PyPDFLoader(file_name)
                all_chunks.extend(text_splitter.split_documents(loader.load()))
            except Exception as e:
                st.sidebar.warning(f"Error loading {file_name}: {e}")
        else:
            st.sidebar.error(f"❌ Missing: {file_name}")
    
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

with st.spinner("Syncing Regulatory Intelligence..."):
    vector_db = load_base_knowledge()

# --- 5. EXECUTION ENGINE ---
uploaded_file = st.file_uploader("Upload YOUR Device Technical Documentation", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Executing Strict Audit..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_loader.load())
        
        # Capture user evidence separately for strict evaluation
        user_text = "\n\n".join([c.page_content for c in user_chunks])
        
        if st.button("🚀 Run Comprehensive Audit"):
            # A. Retrieve Regulatory Context
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14 requirements", k=5)])
            
            # B. The Strict Auditor Prompt (The "What is wrong")
            audit_prompt = f
