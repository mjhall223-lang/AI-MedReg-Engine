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
st.subheader("Automated Gap Analysis vs. EU AI Act & IVDR (2026)")

# --- 1. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' missing in Secrets.")
    st.stop()

# --- 2. INITIALIZE BRAIN ---
try:
    llm = ChatGroq(
        temperature=0, # Zero for audit accuracy
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ LLM ERROR: {e}")
    st.stop()

# --- 3. LOAD CORE REGULATIONS ---
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
                st.sidebar.warning(f"Error reading {file_name}: {e}")
        else:
            st.sidebar.error(f"❌ Missing: {file_name}")
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

with st.spinner("Synchronizing 2026 Regulatory Knowledge Base..."):
    vector_db = load_base_knowledge()
    if vector_db:
        st.sidebar.success("✅ Knowledge Base Online")

# --- 4. UPLOAD & ANALYSIS ---
uploaded_file = st.file_uploader("Upload Technical Documentation (PDF)", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Calculating Compliance Metrics..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_loader.load())
        vector_db.add_documents(user_chunks)
        
        # --- THE SCORECARD ENGINE ---
        if st.button("🚀 Generate Compliance Scorecard"):
            # Retrieval of context (User Doc + Regulations)
            query = "Evaluate Data Governance (Art 10), Transparency (Art 13), Human Oversight (Art 14), and IVDR Transition."
            search_results = vector_db.similarity_search(query, k=10)
            context = "\n\n".join([d.page_content for d in search_results])
            
            # The "Judge" Prompt
            score_prompt = f"""
            SYSTEM: You are a Senior Regulatory Auditor. 
            Score the device based on the provided CONTEXT.
            
            OUTPUT FORMAT (MUST BE EXACT):
            [ART_10_SCORE]: X/10
            [ART_13_SCORE]: X/10
            [ART_14_SCORE]: X/10
            [IVDR_SCORE]: X/10
            [SUMMARY]: Detailed audit report...

            SCORING CRITERIA:
            0-3: Critical Gaps (Non-compliant)
            4-7: Partial Compliance (Action required)
            8-10: High Compliance (Ready for audit)

            CONTEXT:
            {context}
            """
            
            response = llm.invoke(score_prompt).content
            
            # Extract scores using Regex
            def get_score(pattern, text):
                match = re.search(pattern, text)
                return match.group(1) if match else "0"

            s10 = get_score(r"\[ART_10_SCORE\]: (\d+)", response)
            s13 = get_score(r"\[ART_13_SCORE\]: (\d+)", response)
            s14 = get_score(r"\[ART_14_SCORE\]: (\d+)", response)
            sivdr = get_score(r"\[IVDR_SCORE\]: (\d+)", response)

            # --- DISPLAY THE SCORECARD ---
            st.markdown("### 🏆 EXECUTIVE SCORECARD")
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Art 10: Data", f"{s10}/10", delta="High Risk" if int(s10) < 5 else "Moderate", delta_color="inverse")
            col2.metric("Art 13: Transp.", f"{s13}/10")
            col3.metric("Art 14: Oversight", f"{s14}/10")
            col4.metric("IVDR Transition", f"{sivdr}/10")

            st.markdown("---")
            st.markdown("### 📋 AUDITOR'S GAP ANALYSIS")
            # Extract summary part
            summary_part = response.split("[SUMMARY]:")[-1]
            st.write(summary_part)
            
            os.remove(tmp_path)

# Sidebar
with st.sidebar:
    st.markdown("### 🛡️ REGULATORY SHIELD")
    st.info("Version: 1.1.0-Scorecard")
    st.write("**Specialist:** MJ Hall (Bio-AI)")
