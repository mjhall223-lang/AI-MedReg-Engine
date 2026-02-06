import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Page & Branding
st.set_page_config(page_title="AI-MedReg Auditor", page_icon="🛡️", layout="wide")
st.title("🛡️ AI-MedReg Auditor: Professional Edition")
st.markdown("### 2026 Compliance Engine | Gap Analysis & Audit Preparation")

# 2. Security Check
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("🚨 Configuration Error: API Token missing in Streamlit Secrets.")
    st.stop()

# 3. Initialize Reasoning Brain
@st.cache_resource
def load_system():
    # Mistral-7B provides the suggestions and reasoning
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.2,
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_system()

# 4. Audit Engine Logic
def run_audit_engine(uploaded_files):
    all_docs = []
    # Check all possible case variations for your master files
    master_files = ["EU regulations.pdf", "IVDR.pdf", "ivdr.pdf", "Ivdr.pdf", "ivdr.pdf.pdf"]
    
    for mf in master_files:
        if os.path.exists(mf):
            try:
                loader = PyPDFLoader(mf)
                all_docs.extend(loader.load())
            except: continue
    
    for f in uploaded_files:
        temp = f"temp_{f.name}"
        with open(temp, "wb") as buffer:
            buffer.write(f.getbuffer())
        try:
            loader = PyPDFLoader(temp)
            all_docs.extend(loader.load())
        finally:
            if os.path.exists(temp): os.remove(temp)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)
    return FAISS.from_documents(chunks, embeddings)

# 5. The Interface
st.sidebar.header("📁 Client Data Ingestion")
files = st.sidebar.file_uploader("Upload Client Technical File (PDF)", type="pdf", accept_multiple_files=True)

if files:
    with st.spinner("🔄 Building Regulatory Context..."):
        try:
            vector_db = run_audit_engine(files)
            st.sidebar.success("🚀 Audit Engine Online")
            
            st.markdown("---")
            st.subheader("🕵️ Professional Compliance Probe")
            query = st.text_area("What would you like to audit?", 
                                placeholder="e.g., Conduct a gap analysis for Article 10 and suggest specific remediation steps for my tech file.")
            
            if st.button("🔥 RUN NUCLEAR AUDIT"):
                if query:
                    with st.spinner("🧠 Analyzing Gaps & Drafting Suggestions..."):
                        # This chain combines the law with the client file to generate a report
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=vector_db.as_retriever(search_kwargs={"k": 5})
                        )
                        response = qa_chain.invoke(query)
                        
                        st.success("✅ Audit Complete")
                        
                        # Part 1: The Suggestions / Gap Analysis
                        st.markdown("#### 📝 Consultant Report & Remediation Suggestions")
                        st.write(response["result"])
                        
                        # Part 2: The Evidence
                        with st.expander("📚 View Regulatory Evidence (Raw Data)"):
                            evidence = vector_db.similarity_search(query, k=3)
                            for i, d in enumerate(evidence):
                                st.info(f"**Source: {d.metadata.get('source')}**\n\n{d.page_content}")
                else:
                    st.warning("Please enter an audit objective.")
        except Exception as e:
            st.error(f"❌ Engine Error: {e}")
else:
    st.info("👋 Ready to Audit. Please upload a Technical File in the sidebar to begin.")
