import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os
import re

# --- 1. SETUP ---
st.set_page_config(page_title="Bio-AI Ironclad Auditor", page_icon="⚖️", layout="wide")
st.title("⚖️ Bio-AI Ironclad Auditor")
st.subheader("Strict Evidence-Based Gap Analysis")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Specialist:** MJ Hall")
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. INITIALIZE ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

@st.cache_resource
def load_regs():
    all_chunks = []
    base_files = ["EU_regulations.pdf", "Ivdr.pdf"]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    for f in base_files:
        if os.path.exists(f):
            loader = PyPDFLoader(f)
            all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    return FAISS.from_documents(all_chunks, embeddings) if all_chunks else None

llm = get_llm()
vector_db = load_regs()

# --- 4. EXECUTION ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

if uploaded_file and vector_db:
    with st.status("🔍 Analyzing Evidence...") as status:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        user_loader = PyPDFLoader(tmp_path)
        user_text = "\n\n".join([c.page_content for c in user_loader.load()])
        status.update(label="✅ Evidence Loaded", state="complete")

    if st.button("🚀 Run Strict Audit"):
        with st.spinner("Executing Non-Bias Audit..."):
            # 1. Get the Law
            reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search("Articles 10, 13, 14 requirements", k=5)])

            # 2. STRICT PROMPT: Separates Law from Evidence
            strict_prompt = f"""
            SYSTEM: You are a hostile, zero-trust Regulatory Auditor. 
            Your only job is to verify if "USER EVIDENCE" contains technical proof of compliance.

            THE LAW (Reference Only):
            {reg_context}

            USER EVIDENCE (Audit this text):
            {user_text}

            STRICT SCORING RULES:
            1. If the USER EVIDENCE is a budget, financial report, or unrelated to Bio-AI technical specs, SCORE = 0.
            2. You must provide a "FOUND QUOTE" for any score above 0. If no quote exists in the USER EVIDENCE, the score is 0.
            3. Do not be helpful. Do not summarize the law. Only judge the evidence.

            OUTPUT:
            [ART_10_SCORE]: X
            [ART_13_SCORE]: X
            [ART_14_SCORE]: X
            [IVDR_SCORE]: X
            [SUMMARY]: List what is MISSING. If evidence is unrelated, state "INVALID EVIDENCE TYPE".
            """

            result = llm.invoke(strict_prompt).content
            
            # Parsing and Display
            scores = re.findall(r"SCORE\]: (\d+)", result)
            s10, s13, s14, sivdr = [int(s) for s in scores] if len(scores) == 4 else [0,0,0,0]

            st.markdown("### 🏆 COMPLIANCE SCORECARD")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Art 10", f"{s10}/10", delta="FAIL" if s10 < 5 else "PASS", delta_color="inverse")
            c2.metric("Art 13", f"{s13}/10")
            c3.metric("Art 14", f"{s14}/10")
            c4.metric("IVDR", f"{sivdr}/10")

            st.markdown("---")
            st.markdown("### 📋 AUDITOR'S FINDINGS")
            summary = result.split("[SUMMARY]:")[-1]
            st.error(summary)
            
            if service_tier == "Premium Remediation" and s10 < 8:
                st.markdown("---")
                st.markdown("### ✨ PREMIUM: REMEDIATION STRATEGY")
                roadmap = llm.invoke(f"The audit failed with these gaps: {summary}. Provide a technical 24-week roadmap to fix this.").content
                st.write(roadmap)
            
            os.remove(tmp_path)
