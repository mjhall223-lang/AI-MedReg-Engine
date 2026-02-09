import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os
import re

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Bio-AI Compliance Dashboard", page_icon="⚖️", layout="wide")

st.title("⚖️ Bio-AI Compliance & Remediation Engine")
st.subheader("Official Gap Analysis: EU AI Act & IVDR (2026)")
st.markdown("---")

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("## 🛡️ REGULATORY SHIELD")
    st.markdown("**Lead Specialist:** MJ Hall")
    st.info("System Status: v1.3.2 - Stable")
    
    st.markdown("---")
    st.markdown("### 💼 SERVICE LEVEL")
    service_tier = st.radio(
        "Select Your Analysis Tier:", 
        ["Standard Gap Analysis", "Premium Remediation (Consulting)"]
    )
    
    if service_tier == "Premium Remediation (Consulting)":
        st.success("✨ PREMIUM MODE ACTIVE")
    else:
        st.warning("Standard Mode Active")

# --- 3. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' not found in Streamlit Secrets.")
    st.stop()

# --- 4. INITIALIZE BRAIN ---
try:
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ CONNECTION ERROR: {e}")
    st.stop()

# --- 5. LOAD CORE KNOWLEDGE BASE ---
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
                docs = loader.load()
                all_chunks.extend(text_splitter.split_documents(docs))
            except Exception as e:
                st.sidebar.warning(f"Error loading {file_name}: {e}")
        else:
            st.sidebar.error(f"❌ Missing Core File: {file_name}")
    
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

with st.spinner("Syncing 2026 Regulatory Intelligence..."):
    vector_db = load_base_knowledge()

# --- 6. EXECUTION ENGINE ---
uploaded_file = st.file_uploader("Upload Technical Documentation (PDF)", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Executing Strict Audit..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_loader.load())
        user_text = "\n\n".join([c.page_content for c in user_chunks])
        
        if st.button("🚀 Run Comprehensive Audit"):
            # A. Retrieve Regulatory Context
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14 requirements", k=5)])
            
            # B. THE STRICT AUDITOR PROMPT
            audit_prompt = f"""
            SYSTEM: You are a cynical, strict Regulatory Lead Auditor. 
            Grade strictly against the 2024 EU AI Act (2024/1689) and IVDR (2017/746).
            
            GOLD STANDARD (THE LAW): 
            {reg_context}

            USER PROVIDED EVIDENCE: 
            {user_text}

            SCORING (0-10): 
            - Give a 0 if the evidence is unrelated to medical devices (e.g. budgets).
            - Give a 10 only if the evidence explicitly meets the law requirements.
            
            OUTPUT FORMAT:
            [ART_10_SCORE]: X
            [ART_13_SCORE]: X
            [ART_14_SCORE]: X
            [IVDR_SCORE]: X
            [SUMMARY]: Start with PASS or FAIL. List exact missing technical requirements.
            """
            
            audit_result = llm.invoke(audit_prompt).content
            
            # C. Score Parser (Python 3.13 Fix)
            def parse_score(tag):
                pattern = rf"\[{tag}_SCORE\]: (\d+)"
                match
