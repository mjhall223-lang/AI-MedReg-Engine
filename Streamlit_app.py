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
st.subheader("EU AI Act & IVDR Regulatory Auditor")

# --- 1. THE SECRET CHECKER ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("🛑 KEY ERROR: 'GROQ_API_KEY' not found in Streamlit Secrets.")
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

# --- 3. LOAD CORE REGULATIONS ---
@st.cache_resource
def load_base_knowledge():
    all_chunks = []
    # VERBATIM NAMES: Ensure these match your GitHub exactly
    base_files = ["EU_regulations.pdf", "Ivdr.pdf"]
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    for file_name in base_files:
        if os.path.exists(file_name):
            try:
                loader = PyPDFLoader(file_name)
                docs = loader.load()
                all_chunks.extend(text_splitter.split_documents(docs))
            except Exception as e:
                st.sidebar.warning(f"Error reading {file_name}: {e}")
        else:
            st.sidebar.error(f"❌ File Not Found: {file_name}")
            
    if not all_chunks:
        return None
        
    return FAISS.from_documents(all_chunks, embeddings)

# Sync Knowledge Base
with st.spinner("Synchronizing 2026 EU/IVDR Regulations..."):
    vector_db = load_base_knowledge()
    if vector_db:
        st.sidebar.success("✅ Knowledge Base Online")

# --- 4. USER DOC UPLOAD ---
st.markdown("---")
uploaded_file = st.file_uploader("Upload YOUR Device Technical Documentation", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Analyzing Gaps..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        user_loader = PyPDFLoader(tmp_path)
        user_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(user_loader.load())

        # Merge user docs into the regulatory knowledge base
        vector_db.add_documents(user_chunks)
        
        st.success(f"✅ Comparison Engine Ready. Processing {len(user_chunks)} segments.")

        query = st.text_input(
            "Enter Audit Focus:", 
            value="Perform a gap analysis between this device and EU AI Act Articles 10 & 14 and IVDR transition requirements."
        )

        if st.button("Run Audit"):
            search_results = vector_db.similarity_search(query, k=8)
            context = "\n\n".join([d.page_content for d in search_results])
            
            prompt = f"""
            SYSTEM: You are a Lead AI Regulatory Auditor for Medical Devices. 
            Identify critical gaps between the CONTEXT (User Doc + Regs) and 2026 mandates.
            Highlight specific failures regarding Article 10 and Article 14.
            FORMAT: Use 🔴 RED, 🟡 YELLOW, and 🟢 GREEN headers.

            CONTEXT:
            {context}

            QUESTION: {query}
            """
            
            response = llm.invoke(prompt)
            st.markdown("---")
            st.markdown("### 📋 OFFICIAL GAP ANALYSIS REPORT")
            st.write(response.content)
            os.remove(tmp_path)

with st.sidebar:
    st.markdown("### 🛡️ REGULATORY SHIELD")
    st.info("Version: 1.0.5-Stable")
    st.write("**Specialist:** MJ Hall (Bio-AI)")
