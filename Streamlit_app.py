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
st.set_page_config(page_title="Bio-AI Compliance Scorecard", page_icon="📊", layout="wide")
st.title("📊 Bio-AI Compliance Scorecard")
st.subheader("Executive Gap Analysis vs. EU AI Act & IVDR (2026)")

# --- 1. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' not found in Secrets.")
    st.stop()

# --- 2. INITIALIZE BRAIN ---
try:
    llm = ChatGroq(
        temperature=0, # Absolute precision for auditing
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ CONNECTION ERROR: {e}")
    st.stop()

# --- 3. LOAD CORE KNOWLEDGE BASE ---
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

with st.spinner("Syncing 2026 Regulatory Intelligence..."):
    vector_db = load_base_knowledge()
    if vector_db:
        st.sidebar.success("✅ Knowledge Base Online")

# --- 4. SCORECARD ENGINE ---
uploaded_file = st.file_uploader("Upload YOUR Device Technical Documentation", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Executing Audit..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_loader.load())
        vector_db.add_documents(user_chunks)
        
        st.info("💡 Files Merged. Ready for Scoring.")

        if st.button("🚀 Run Full Compliance Audit"):
            # Retrieval of context (User Doc + Regulations)
            query = "Evaluate Data Governance (Art 10), Transparency (Art 13), Human Oversight (Art 14), and IVDR Transition Requirements."
            search_results = vector_db.similarity_search(query, k=10)
            context = "\n\n".join([d.page_content for d in search_results])
            
            # The Auditor Prompt
            audit_prompt = f"""
            SYSTEM: You are a Lead Regulatory Auditor. You must score the device based ON THE PROVIDED CONTEXT.
            
            SCORING (0-10):
            0-3: Critical/Missing | 4-7: Incomplete | 8-10: Compliant
            
            OUTPUT FORMAT (MANDATORY):
            [ART_10_SCORE]: X
            [ART_13_SCORE]: X
            [ART_14_SCORE]: X
            [IVDR_SCORE]: X
            [SUMMARY]: Detailed audit report...

            CONTEXT:
            {context}
            """
            
            result = llm.invoke(audit_prompt).content
            
            # Extract scores with Regex
            def parse_score(tag):
                match = re.search(f"\\{tag}\\]: (\\d+)", result)
                return int(match.group(1)) if match else 0

            s10, s13, s14, sivdr = parse_score("ART_10"), parse_score("ART_13"), parse_score("ART_14"), parse_score("IVDR")

            # --- DISPLAY DASHBOARD ---
            st.markdown("### 🏆 COMPLIANCE SCORECARD")
            
            m1, m2, m3, m4 = st.columns(4)
            
            m1.metric("Art 10: Data", f"{s10}/10", delta="Warning" if s10 < 7 else "Passed", delta_color="inverse" if s10 < 7 else "normal")
            m2.metric("Art 13: Transp.", f"{s13}/10")
            m3.metric("Art 14: Oversight", f"{s14}/10")
            m4.metric("IVDR Status", f"{sivdr}/10")

            st.markdown("---")
            st.markdown("### 📋 AUDITOR'S DETAILED FINDINGS")
            summary = result.split("[SUMMARY]:")[-1]
            st.markdown(summary)
            
            os.remove(tmp_path)

# Sidebar Branding
with st.sidebar:
    st.markdown("### 🛡️ REGULATORY SHIELD")
    st.info("Ver: 1.1.0 (Scorecard Edition)")
    st.write("**Specialist:** MJ Hall (Bio-AI)")
