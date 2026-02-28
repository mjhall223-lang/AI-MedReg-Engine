import streamlit as st
import os
import re
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETUP ---
st.set_page_config(page_title="Federal Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR (AUDIT SELECTOR) ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    # THE KEY CHANGE: Added framework selection
    audit_choice = st.selectbox(
        "Choose Audit Framework", 
        ["EU Medical", "Colorado AI Act", "CMMC 2.0"]
    )
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# Mapping choices to your new GitHub folders
folder_map = {
    "EU Medical": "Regulations/EU_Medical/", # This handles your original PDFs
    "Colorado AI Act": "Regulations/Colorado/",
    "CMMC 2.0": "Regulations/CMMC/"
}
selected_path = folder_map[audit_choice]

# --- 3. DYNAMIC DOCUMENT LOADER ---
@st.cache_resource
def load_regs(path):
    all_chunks = []
    # Key fix: This now loops through whatever folder you picked in the sidebar
    if os.path.exists(path):
        for filename in os.listdir(path):
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, filename))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

# Initialize LLM and DB
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
vector_db = load_regs(selected_path)

# --- 4. EXECUTION ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

if uploaded_file and vector_db:
    with st.status("🔍 Analyzing Evidence...") as status:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        user_loader = PyPDFLoader(tmp_path)
        user_text = "\n\n".join([c.page_content for c in user_loader.load()])
        status.update(label=f"✅ Evidence Loaded for {audit_choice}", state="complete")

    if st.button("🚀 Run Strict Audit"):
        with st.spinner("Comparing Evidence to Law..."):
            # Retrieval: Get specific context for the chosen audit
            search_query = "Impact assessment bias" if "Colorado" in audit_choice else "Access control"
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])

            # STRICT PROMPT (Now supports Colorado Investment Evidence)
            strict_prompt = f"""
            SYSTEM: You are a zero-trust Regulatory Auditor for {audit_choice}.
            
            RULES:
            1. If auditing Colorado AI Act, evidence of AI stocks (NVDA, PLTR, etc.) is VALID for transparency checks.
            2. For CMMC, look for data security and encryption.
            3. If evidence is unrelated to {audit_choice}, SCORE = 0.

            THE LAW: {reg_context}
            USER EVIDENCE: {user_text}

            OUTPUT:
            [SCORE_1]: X (Transparency/Access)
            [SCORE_2]: X (Bias/Security)
            [SUMMARY]: List MISSING items.
            """

            result = llm.invoke(strict_prompt).content
            st.markdown("### 🏆 AUDIT FINDINGS")
            st.write(result)
