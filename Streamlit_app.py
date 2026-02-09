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
    st.info("Version: 1.2.6 (Premium Edition)")
    st.markdown("---")
    st.markdown("### 💼 SERVICE LEVEL")
    service_tier = st.radio(
        "Select Analysis Level:", 
        ["Standard Gap Analysis", "Premium Remediation (Consulting)"]
    )
    if service_tier == "Premium Remediation (Consulting)":
        st.success("✨ Premium Mode Enabled")
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
        user_docs = user_loader.load()
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_docs)
        user_text = "\n\n".join([c.page_content for c in user_chunks])
        
        if st.button("🚀 Run Comprehensive Audit"):
            # A. Retrieve Regulatory Context
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14 requirements", k=5)])
            
            # B. The Strict Auditor Prompt
            audit_prompt = f"""
            SYSTEM: You are a cynical, strict Regulatory Lead Auditor. 
            Grade the device based ONLY on the provided evidence.
            
            GOLD STANDARD (LAW):
            {reg_context}

            USER EVIDENCE (THE DOCUMENT):
            {user_text}

            SCORING (0-10):
            If the evidence is unrelated to medical devices, give 0.
            
            OUTPUT FORMAT:
            [ART_10_SCORE]: X
            [ART_13_SCORE]: X
            [ART_14_SCORE]: X
            [IVDR_SCORE]: X
            [SUMMARY]: Start with PASS or FAIL. List exact missing technical requirements.
            """
            
            audit_result = llm.invoke(audit_prompt).content
            
            # C. Parsing Scores
            def parse_score(tag):
                pattern = rf"\[{tag}_SCORE\]: (\d+)"
                match = re.search(pattern, audit_result)
                return int(match.group(1)) if match else 0

            s10, s13, s14, sivdr = parse_score("ART_10"), parse_score("ART_13"), parse_score("ART_14"), parse_score("IVDR")

            # --- DISPLAY DASHBOARD ---
            st.markdown("### 🏆 COMPLIANCE SCORECARD")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Art 10: Data", f"{s10}/10")
            m2.metric("Art 13: Transp.", f"{s13}/10")
            m3.metric("Art 14: Oversight", f"{s14}/10")
            m4.metric("IVDR Status", f"{sivdr}/10")

            st.markdown("---")
            st.markdown("### 📋 AUDITOR'S FINDINGS")
            summary = audit_result.split("[SUMMARY]:")[-1]
            if "FAIL" in summary:
                st.error(summary)
            else:
                st.success(summary)

            # D. The Consultant Prompt (PREMIUM ONLY)
            if service_tier == "Premium Remediation (Consulting)":
                st.markdown("---")
                st.markdown("### ✨ PREMIUM: STRATEGIC REMEDIATION ROADMAP")
                
                
                consultant_prompt = f"""
                SYSTEM: You are a high-priced Bio-AI Regulatory Consultant. 
                Based on these FAILURES: {summary}
                Provide a strategic roadmap to reach 10/10 compliance.
                Include technical documentation requirements and Article 14 testing protocols.
                """
                
                with st.spinner("Generating Strategic Roadmap..."):
                    suggestions = llm.invoke(consultant_prompt).content
                    st.write(suggestions)
                    st.info("💡 Want MJ Hall to implement this strategy? Contact for partnership.")
            
            os.remove(tmp_path)
