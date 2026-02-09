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
st.set_page_config(page_title="Bio-AI Strict Auditor", page_icon="⚖️", layout="wide")
st.title("⚖️ Bio-AI Strict Compliance Auditor")
st.subheader("No-Nonsense Gap Analysis (v1.2.0)")

# --- 1. INITIALIZE ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 Missing GROQ_API_KEY")
    st.stop()

try:
    llm = ChatGroq(
        temperature=0, # Maximum strictness
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ Error: {e}")
    st.stop()

# --- 2. LOAD REGS ---
@st.cache_resource
def load_base_regs():
    all_chunks = []
    base_files = ["EU_regulations.pdf", "Ivdr.pdf"]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    for f in base_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

vector_db = load_base_regs()

# --- 3. AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Device Technical Documentation", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Analyzing Evidence..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(user_loader.load())
        
        # We store the user chunks separately to tell the AI what is "Evidence"
        user_text = "\n\n".join([c.page_content for c in user_chunks])
        
        if st.button("🚀 Run Strict Audit"):
            # Search regs for context
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14 requirements", k=5)])

            # --- THE STRICT AUDITOR PROMPT ---
            strict_prompt = f"""
            SYSTEM: You are a cynical, strict Regulatory Lead Auditor. 
            Your goal is to find reasons to FAIL this device.
            
            GOLD STANDARD (THE LAW):
            {reg_context}

            USER PROVIDED EVIDENCE (THE DOCUMENT):
            {user_text}

            INSTRUCTIONS:
            1. Only use the "USER PROVIDED EVIDENCE" to judge compliance. 
            2. If the user evidence is unrelated to medical devices (e.g., a budget, a grocery list, or general business info), you MUST give a score of 0 for all categories.
            3. Do NOT summarize the law. Only identify what is MISSING from the user evidence.
            4. Be extremely harsh. "Maybe" or "Implied" = 0.

            OUTPUT FORMAT:
            [ART_10_SCORE]: X
            [ART_13_SCORE]: X
            [ART_14_SCORE]: X
            [IVDR_SCORE]: X
            [SUMMARY]: Start with "PASS" or "FAIL". List exact missing technical requirements.
            """

            result = llm.invoke(strict_prompt).content
            
            # Parsing and Display
            scores = re.findall(r"SCORE\]: (\d+)", result)
            s10, s13, s14, sivdr = scores if len(scores) == 4 else [0,0,0,0]

            st.markdown("### 🏆 COMPLIANCE SCORECARD")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Art 10", f"{s10}/10")
            c2.metric("Art 13", f"{s13}/10")
            c3.metric("Art 14", f"{s14}/10")
            c4.metric("IVDR", f"{sivdr}/10")

            st.markdown("---")
            st.markdown("### 📋 AUDITOR'S FINDINGS")
            st.write(result.split("[SUMMARY]:")[-1])
            os.remove(tmp_path)
