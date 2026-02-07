import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Bio-AI Compliance Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ Bio-AI Compliance Engine")
st.subheader("EU AI Act Regulatory Auditor (Articles 10 & 14)")

# --- 1. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' not found in Streamlit Secrets.")
    st.info("Please go to App Settings > Secrets and add: GROQ_API_KEY = 'your_key_here'")
    st.stop()

# --- 2. INITIALIZE BRAIN ---
try:
    llm = ChatGroq(
        temperature=0.1, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )
except Exception as e:
    st.error(f"⚠️ LLM CONNECTION ERROR: {e}")
    st.stop()

# --- 3. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Technical Documentation (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing Clinical Data Structure..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        st.success(f"✅ Audit Engine Online: {len(chunks)} clinical segments indexed.")

        query = st.text_input(
            "Enter Audit Focus:", 
            value="Audit for Article 10.3 (Data bias) and Article 14 (Human oversight)."
        )

        if st.button("Run Audit"):
            search_results = vector_db.similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in search_results])
            
            prompt = f"""
            SYSTEM: You are a Lead AI Regulatory Auditor for Medical Devices. 
            Ground every answer in Article 10 (Data Governance) and Article 14 (Human Oversight).
            FORMAT: Use 🔴 RED, 🟡 YELLOW, and 🟢 GREEN headers for the report.

            CONTEXT:
            {context}

            QUESTION: {query}
            """
            
            response = llm.invoke(prompt)
            st.markdown("---")
            st.markdown("### 📋 OFFICIAL GAP ANALYSIS REPORT")
            st.write(response.content)
            
            # Sidebar Disclaimer (Your "Legal Shield")
            with st.sidebar:
                st.markdown("### 🛡️ REGULATORY SHIELD")
                st.info("This tool provides a gap analysis based on 2026 mandates. It is not legal advice.")
                st.write(f"**Version:** 1.0.4-Stable")
                st.write(f"**Specialist:** MJ Hall (Bio-AI)")
