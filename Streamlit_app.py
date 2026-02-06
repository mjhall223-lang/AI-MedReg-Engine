import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Page Configuration
st.set_page_config(page_title="AI-MedReg Auditor", page_icon="🛡️")
st.title("🛡️ AI-MedReg Auditor")
st.markdown("### 2026 EU AI Act & IVDR Compliance Engine")

# Security Check: Secrets
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("🚨 API Token missing! Go to Settings > Secrets and add 'HUGGINGFACEHUB_API_TOKEN'.")
    st.stop()

# 2. Setup the Brain
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

def build_engine(uploaded_files):
    all_docs = []
    
    # MASTER REGULATIONS: Checking all possible naming variations to prevent 'File Not Found'
    possible_names = ["EU regulations.pdf", "IVDR.pdf", "ivdr.pdf", "Ivdr.pdf", "ivdr.pdf.pdf"]
    
    for mf in possible_names:
        if os.path.exists(mf):
            try:
                loader = PyPDFLoader(mf)
                all_docs.extend(loader.load())
            except Exception:
                continue
    
    # CLIENT DOCUMENTS: Temporary processing
    for uploaded_file in uploaded_files:
        temp_name = f"temp_{uploaded_file.name}"
        with open(temp_name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            loader = PyPDFLoader(temp_name)
            all_docs.extend(loader.load())
        finally:
            if os.path.exists(temp_name):
                os.remove(temp_name)

    if not all_docs:
        raise ValueError("No documents found. Check if the PDFs are in your GitHub folder.")

    # Nuclear Splitting & Indexing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(all_docs)
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db, len(chunks)

# 3. User Interface
files = st.file_uploader("Upload Tech Files for Audit (PDF)", type="pdf", accept_multiple_files=True)

if files:
    with st.spinner("🔄 Syncing Master Laws + Tech Files..."):
        try:
            db, count = build_engine(files)
            st.success(f"🚀 Audit Engine Online: {count} data segments indexed.")
            
            query = st.text_input("Enter Compliance Probe (e.g. 'Audit Article 10.3 gaps')")
            
            if st.button("🔥 RUN NUCLEAR AUDIT"):
                if query:
                    with st.spinner("🕵️ Scanning Regulatory Gaps..."):
                        docs = db.similarity_search(query, k=4)
                        st.subheader("🕵️ Audit Findings")
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get('source', 'Unknown File')
                            st.info(f"**Evidence {i+1} (Source: {source})**\n\n{doc.page_content}")
                else:
                    st.warning("Please enter a probe question first.")
        except Exception as e:
            st.error(f"❌ Engine Error: {e}")
