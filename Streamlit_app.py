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

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ REGULATORY SHIELD")
    st.markdown("**Lead Specialist:** MJ Hall")
    service_tier = st.radio("Analysis Tier:", ["Standard Gap Analysis", "Premium Remediation (Consulting)"])
    st.info("System Status: v1.3.3 - Debug Mode")

# --- 3. INITIALIZE BRAIN ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 Missing GROQ_API_KEY in Secrets.")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )

llm = get_llm()

# --- 4. LOAD CORE REGS ---
@st.cache_resource
def load_base_knowledge():
    all_chunks = []
    base_files = ["EU_regulations.pdf", "Ivdr.pdf"]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    for file_name in base_files:
        if os.path.exists(file_name):
            loader = PyPDFLoader(file_name)
            docs = loader.load()
            all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(docs))
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

vector_db = load_base_knowledge()

# --- 5. EXECUTION ENGINE ---
uploaded_file = st.file_uploader("Upload Technical Documentation (PDF)", type="pdf")

if uploaded_file and vector_db:
    # We use st.status to show progress so the user knows it hasn't "stopped"
    with st.status("🔍 Processing Document...", expanded=True) as status:
        st.write("1. Creating temporary workspace...")
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        st.write("2. Extracting technical evidence...")
        user_loader = PyPDFLoader(tmp_path)
        user_docs = user_loader.load()
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_docs)
        
        # Safeguard: Truncate text if it's massive to prevent token crashes
        user_text = "\n\n".join([c.page_content for c in user_chunks])
        if len(user_text) > 40000:
            user_text = user_text[:40000] + "\n\n[TEXT TRUNCATED FOR PERFORMANCE]"
        
        st.write("3. Document ready for audit.")
        status.update(label="✅ Document Indexed", state="complete", expanded=False)

    if st.button("🚀 Run Comprehensive Audit"):
        with st.status("⚖️ Executing AI Audit...", expanded=True) as audit_status:
            try:
                st.write("Phase A: Retrieving Regulatory Standards...")
                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14", k=5)])
                
                st.write("Phase B: Analyzing Gaps with Llama-3.3...")
                audit_prompt = f"""
                SYSTEM: You are a strict Regulatory Auditor. Grade strictly.
                LAW: {reg_context}
                EVIDENCE: {user_text}
                
                OUTPUT FORMAT:
                [ART_10_SCORE]: X
                [ART_13_SCORE]: X
                [ART_14_SCORE]: X
                [IVDR_SCORE]: X
                [SUMMARY]: Detailed findings...
                """
                
                # Using a try/except specifically for the API call
                response = llm.invoke(audit_prompt)
                audit_result = response.content
                
                st.write("Phase C: Parsing Scores...")
                def parse_score(tag):
                    pattern = rf"\[{tag}_SCORE\]: (\d+)"
                    match = re.search(pattern, audit_result)
                    return int(match.group(1)) if match else 0

                s10, s13, s14, sivdr = parse_score("ART_10"), parse_score("ART_13"), parse_score("ART_14"), parse_score("IVDR")

                # DISPLAY RESULTS
                st.markdown("### 🏆 COMPLIANCE SCORECARD")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Art 10", f"{s10}/10")
                m2.metric("Art 13", f"{s13}/10")
                m3.metric("Art 14", f"{s14}/10")
                m4.metric("IVDR", f"{sivdr}/10")

                st.markdown("---")
                st.markdown("### 📋 AUDITOR'S FINDINGS")
                summary = audit_result.split("[SUMMARY]:")[-1]
                st.error(summary) if s10 < 5 else st.success(summary)

                if service_tier == "Premium Remediation (Consulting)":
                    st.write("Phase D: Generating Strategic Roadmap...")
                    consultant_prompt = f"As a consultant, provide a 24-week roadmap to fix: {summary}"
                    roadmap = llm.invoke(consultant_prompt).content
                    st.markdown("---")
                    st.markdown("### ✨ PREMIUM: REMEDIATION STRATEGY")
                    st.write(roadmap)

                audit_status.update(label="✅ Audit Complete", state="complete")

            except Exception as e:
                st.error(f"⚠️ Audit Failed: {e}")
                audit_status.update(label="❌ Audit Failed", state="error")
            
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
