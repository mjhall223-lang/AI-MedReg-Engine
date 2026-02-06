import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="Bio-AI Compliance Engine", page_icon="🛡️")
st.title("🛡️ Bio-AI Compliance Engine")
st.markdown("> **Target:** EU AI Act (August 2026 Mandate)")

# --- 1. INITIALIZE BRAIN (Using Secrets) ---
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-specdec", api_key=groq_api_key)
except Exception as e:
    st.error("Missing GROQ_API_KEY in Streamlit Secrets!")
    st.stop()

# --- 2. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload Clinical Tech Documentation (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing Documentation..."):
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load & Split
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)

        # Create Vector DB (Fast local embeddings)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        st.success(f"Audit Engine Online: {len(chunks)} segments indexed.")

        # --- 3. AUDIT CONSOLE ---
        query = st.text_input("Enter Audit Question (e.g., 'Check Article 10.3 compliance')", 
                             value="Audit this summary against Article 10 bias and Article 14 human oversight.")

        if st.button("Run Audit"):
            # Retrieval
            search_results = vector_db.similarity_search(query, k=5)
            context = "\n\n".join([d.page_content for d in search_results])
            
            # Prompting
            prompt = f"""
            SYSTEM: You are a Lead AI Regulatory Auditor for Medical Devices. 
            Ground every answer in Article 10 (Data Governance) and Article 14 (Human Oversight).
            FORMAT: 🔴 RED (Fail), 🟡 YELLOW (Warning), 🟢 GREEN (Pass).

            CONTEXT:
            {context}

            QUESTION: {query}
            """
            
            response = llm.invoke(prompt)
            st.markdown("### 📋 AUDIT REPORT")
            st.write(response.content)
