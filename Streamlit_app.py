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

# --- 3. LOAD CORE REGULATIONS (Pre-set Files) ---
@st.cache_resource
def load_base_knowledge():
    all_chunks = []
    # Verbatim names from your GitHub repository
    base_files = ["EU regulations.pdf", "Ivdr.pdf"]
    
    loader_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    for file_name in base_files:
        if os.path.exists(file_name):
            try:
                loader = PyPDFLoader(file_name)
                docs = loader.load()
                all_chunks.extend(text_splitter.split_documents(docs))
            except Exception as e:
                st.warning(f"Could not load {file_name}: {e}")
        else:
            st.error(f"Missing Core File: {file_name} not found in repository.")
            
    if not all_chunks:
        return None
        
    return FAISS.from_documents(all_chunks, loader_embeddings)

# Initialize the Regulation Knowledge Base
with st.spinner("Synchronizing 2026 EU Regulations..."):
    vector_db = load_base_knowledge()
    if vector_db:
        st.sidebar.success("✅ Base Regulations Loaded")

# --- 4. USER DOCUMENT UPLOAD & ANALYSIS ---
uploaded_file = st.file_uploader("Upload YOUR Device Technical Documentation", type="pdf")

if uploaded_file and vector_db:
    with st.spinner("Merging Documentation with Regulatory Context..."):
        # Save upload to temp file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load and chunk user doc
        user_loader = PyPDFLoader(tmp_path)
        user_docs = user_loader.load()
        user_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        user_chunks = user_splitter.split_documents(user_docs)

        # Merge user chunks into our existing Regulation Vector DB
        vector_db.add_documents(user_chunks)
        
        st.success(f"✅ Ready: {len(user_chunks)} user segments added to analysis engine.")

        query = st.text_input(
            "Enter Audit Focus:", 
            value="Perform a gap analysis between this device and EU AI Act Article 10/14 and IVDR transition requirements."
        )

        if st.button("Run Audit"):
            with st.spinner("Analyzing Gaps..."):
                search_results = vector_db.similarity_search(query, k=7)
                context = "\n\n".join([d.page_content for d in search_results])
                
                prompt = f"""
                SYSTEM: You are a Lead AI Regulatory Auditor for Medical Devices (Bio-AI). 
                Compare the provided CONTEXT (User Doc + Regulations) to identify gaps.
                Highlight where the user's device fails to meet Article 10 (Data) and Article 14 (Human Oversight).
                FORMAT: Use 🔴 RED (Critical Gap), 🟡 YELLOW (Warning), and 🟢 GREEN (Compliant) headers.

                CONTEXT:
                {context}

                QUESTION: {query}
                """
                
                response = llm.invoke(prompt)
                st.markdown("---")
                st.markdown("### 📋 OFFICIAL GAP ANALYSIS REPORT")
                st.write(response.content)
                
                # Clean up temp file
                os.remove(tmp_path)

# Sidebar Info
with st.sidebar:
    st.markdown("### 🛡️ REGULATORY SHIELD")
    st.info("Version: 1.0.4-Stable (2026 Compliance Update)")
    st.write("**Core Regs:** EU AI Act, IVDR")
    st.write("**Specialist:** MJ Hall")
