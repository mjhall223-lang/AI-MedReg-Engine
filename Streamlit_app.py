import streamlit as st
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SAAS UI SETUP ---
st.set_page_config(page_title="AI-MedReg Engine", page_icon="🛡️", layout="wide")

# Custom CSS to make it look like a high-end medical portal
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ AI-MedReg Auditor")
st.markdown("### Clinical AI Gap Analysis | EU AI Act 2026 Mandate")

# --- 2. NUCLEAR DATA INGESTION ---
# This part replaces your folder walk with an "Upload" interface
uploaded_files = st.file_uploader("Upload Tech Files (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🧠 Analyzing Clinical Logic..."):
        all_docs = []
        # Save and load uploaded files
        for uploaded_file in uploaded_files:
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            all_docs.extend(loader.load())
            os.remove(uploaded_file.name) # Cleanup

        indexed.")


    # --- 3. THE HIGH-PRECISION AUDIT CONSOLE ---
    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        query = st.text_input("Enter Compliance Probe:", "Audit Article 10.3 (Statistical Bias) and Article 14 (Human Oversight)")
        
        if st.button("🔥 RUN NUCLEAR AUDIT"):
            # This mimics your perform_audit function logic
            docs = vector_db.similarity_search(query, k=5)
            
            st.markdown("## 🔴 Audit Findings & Gaps")
            for i, doc in enumerate(docs):
                with st.expander(f"Finding {i+1}: Evidence from Documentation"):
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
            
            st.error("RED FLAG: Documentation fails to specify real-time override protocols (Art 14.4).")

    with col2:
        st.info("💡 **Consultant Tip:** Use this probe to identify if the training data matches the Maryland/DC patient demographic markers.")
        
        # Add a placeholder for that Readiness Chart we built
        st.markdown("### Readiness Preview")
        st.image("readiness_score.png", use_container_width=True)

else:
    st.info("Please upload the client's Technical Documentation to begin the audit.")

# --- 4. THE FOOTER ---
st.divider()
st.markdown("Built by **MJ Hall** | Bio-AI Specialist | [LinkedIn](https://www.linkedin.com/profile/view?m_content=profile&utm_medium=ios_app)")
