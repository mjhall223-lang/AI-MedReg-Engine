import streamlit as st
import os
import re
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import engine  # Importing your engine file

# --- 1. SETUP ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    audit_framework = st.selectbox(
        "Select Regulatory Framework",
        ["EU AI Act (Medical)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# Mapping selection to folders
framework_folders = {
    "EU AI Act (Medical)": "Regulations/EU_Medical",
    "Colorado AI Act": "Regulations/Colorado",
    "CMMC 2.0 / NIST 800-171": "Regulations/CMMC"
}
selected_reg_path = framework_folders[audit_framework]

# --- 3. INITIALIZE MODELS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

llm = get_llm()

# --- 4. EXECUTION ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

# The button is now OUTSIDE the 'if' block so it is always visible
st.markdown("---")
if st.button("🚀 Run Strict Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 Analyzing...") as status:
            # 1. Process Evidence
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            user_loader = PyPDFLoader(tmp_path)
            user_text = "\n\n".join([c.page_content for c in user_loader.load()])
            
            # 2. Load the correct Regulations
            vector_db = engine.load_knowledge_base(selected_reg_path)
            
            if vector_db:
                # 3. Run Audit
                result = engine.perform_audit(user_text, vector_db, audit_framework, llm)
                status.update(label="✅ Analysis Complete!", state="complete")
                
                # 4. Display Results
                st.markdown(f"### 🏆 {audit_framework} FINDINGS")
                st.write(result)
            else:
                st.error(f"Could not find regulation files in {selected_reg_path}")
