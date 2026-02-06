import streamlit as st
import os

# 1. Force-load the heavy lifting
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
except ImportError as e:
    st.error(f"⚠️ System still installing: {e}. Please wait 2 minutes.")
    st.stop()

# 2. Page Configuration
st.set_page_config(page_title="AI-MedReg Auditor", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-MedReg Auditor: Professional Edition")

# 3. Secret Key Check
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("🚨 Configuration Error: Go to Settings > Secrets and add 'HUGGINGFACEHUB_API_TOKEN'.")
    st.stop()

@st.cache_resource
def initialize_brain():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.2,
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = initialize_brain()

# 4. Processing Engine
def build_engine(uploaded_files):
    docs = []
    # Check for master files
    master_files = ["EU regulations.pdf", "IVDR.pdf", "ivdr.pdf", "Ivdr.pdf"]
    for mf in master_files:
        if os.path.exists(mf):
            try: docs.extend(PyPDFLoader(mf).load())
            except: pass
    
    # Check for user files
    for f in uploaded_files:
        temp = f"temp_{f.name}"
        with open(temp, "wb") as b: b.write(f.getbuffer())
        try: docs.extend(PyPDFLoader(temp).load())
        finally:
            if os.path.exists(temp): os.remove(temp)
            
    if not docs:
        raise ValueError("No documents found to index.")
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return FAISS.from_documents(splitter.split_documents(docs), embeddings)

# 5. The UI
files = st.sidebar.file_uploader("Upload Tech Files (PDF)", type="pdf", accept_multiple_files=True)

if files:
    try:
        db = build_engine(files)
        st.sidebar.success("🚀 Engine Online")
        
        st.subheader("🕵️ Compliance Audit & Suggestions")
        query = st.text_area("Audit Objective:", placeholder="e.g. Provide a gap analysis for Article 10.3 and remediation steps.")
        
        if st.button("🔥 RUN NUCLEAR AUDIT"):
            if query:
                with st.spinner("🧠 Generating Suggestions..."):
                    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 5}))
                    response = qa.invoke(query)
                    st.success("✅ Audit Complete")
                    st.markdown("#### 📝 Report & Prep Suggestions")
                    st.write(response["result"])
            else: st.warning("Please enter an objective.")
    except Exception as e:
        st.error(f"❌ System Error: {e}")
else:
    st.info("👋 Ready to begin. Upload a file in the sidebar to start the audit.")
