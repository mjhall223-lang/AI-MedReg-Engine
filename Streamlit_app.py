import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Setup
st.set_page_config(page_title="AI-MedReg Auditor", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-MedReg Auditor: Professional Edition")
st.markdown("### Gap Analysis & Audit Preparation")

if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("🚨 API Token missing in Secrets.")
    st.stop()

@st.cache_resource
def load_system():
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", temperature=0.2, huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_system()

def run_audit(uploaded_files):
    all_docs = []
    # Checks for all name variations of your master files
    master_files = ["EU regulations.pdf", "IVDR.pdf", "ivdr.pdf", "Ivdr.pdf"]
    for mf in master_files:
        if os.path.exists(mf):
            try:
                all_docs.extend(PyPDFLoader(mf).load())
            except: continue
    for f in uploaded_files:
        temp = f"temp_{f.name}"
        with open(temp, "wb") as buffer: buffer.write(f.getbuffer())
        try:
            all_docs.extend(PyPDFLoader(temp).load())
        finally:
            if os.path.exists(temp): os.remove(temp)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return FAISS.from_documents(splitter.split_documents(all_docs), embeddings)

# 2. UI
files = st.sidebar.file_uploader("Upload Tech Files", type="pdf", accept_multiple_files=True)
if files:
    try:
        vector_db = run_audit(files)
        st.sidebar.success("🚀 Engine Online")
        query = st.text_area("Audit Objective:", placeholder="e.g. Provide a gap analysis for Article 10.")
        if st.button("🔥 RUN NUCLEAR AUDIT"):
            if query:
                with st.spinner("🧠 Generating Suggestions..."):
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(search_kwargs={"k": 5}))
                    response = qa.invoke(query)
                    st.success("✅ Audit Complete")
                    st.markdown("#### 📝 Suggestions")
                    st.write(response["result"])
                    with st.expander("📚 Evidence"):
                        for d in vector_db.similarity_search(query, k=3):
                            st.info(f"**Source: {d.metadata.get('source')}**\n\n{d.page_content}")
            else: st.warning("Enter a probe.")
    except Exception as e: st.error(f"❌ Error: {e}")
else: st.info("👋 Upload a file to start.")
