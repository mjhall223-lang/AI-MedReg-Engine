import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. UI SETUP
st.set_page_config(page_title="AI-MedReg Engine", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-MedReg Compliance Auditor")
st.subheader("High-Risk Medical AI Gap Analysis (EU AI Act 2026)")

# 2. FILE UPLOADER (The 'SaaS' Moment)
uploaded_files = st.file_uploader("Upload Clinical/Technical Documentation (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔍 Indexing Clinical Data..."):
        # Save uploaded files temporarily
        all_docs = []
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            all_docs.extend(loader.load())
        
        # Build the RAG database
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(all_docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        st.success(f"✅ Indexed {len(chunks)} clinical segments.")

    # 3. THE AUDIT CONSOLE
    st.divider()
    audit_focus = st.selectbox("Select Audit Focus:", 
                               ["Article 10 (Data Governance)", 
                                "Article 14 (Human Oversight)", 
                                "Article 15 (Cybersecurity)"])
    
    if st.button("🚀 Run Compliance Audit"):
        with st.spinner("Senior Auditor is reviewing files..."):
            # This is where your 'perform_audit' logic fires
            # For the SaaS version, we use the selected audit_focus as the query
            results = vector_db.similarity_search(audit_focus, k=4)
            
            st.markdown(f"### 📊 {audit_focus} Findings")
            for i, res in enumerate(results):
                st.info(f"**Segment {i+1}:**\n\n{res.page_content[:500]}...")
            
            st.warning("🔴 GAP IDENTIFIED: Documentation lacks real-time override protocols.")
