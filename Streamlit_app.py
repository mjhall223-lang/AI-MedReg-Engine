import streamlit as st
import os
import tempfile
import re
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Specialist:** MJ Hall")
    audit_framework = st.selectbox(
        "Select Regulatory Framework",
        ["EU AI Act (Medical)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. DYNAMIC MAPPING (Matches your GitHub exactly) ---
framework_folders = {
    "EU AI Act (Medical)": ".",  # Looks in main folder for EU_regulations.pdf & lvdr.pdf
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 / NIST 800-171": "Regulations/Regulations/CMMC" # Matches the typo in your folder name
}
selected_reg_path = framework_folders[audit_framework]

# --- 4. CORE FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    
    if not all_chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

# --- 5. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

st.markdown("---")
# Button is always visible so you can see it on your phone!
if st.button("🚀 Run Strict Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 Processing Audit...") as status:
            # Step 1: Handle User File
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            user_loader = PyPDFLoader(tmp_path)
            user_text = "\n\n".join([c.page_content for c in user_loader.load()])
            
            # Step 2: Load Regulations
            vector_db = load_knowledge_base(selected_reg_path)
            
            if vector_db:
                # Step 3: Setup Auditor Personality & Search
                if "Colorado" in audit_framework:
                    system_role = "Colorado AI Act Auditor. AI stock holdings (NVDA, PLTR) are VALID evidence for transparency and business risk audits."
                    search_query = "Impact assessment bias algorithmic discrimination duty of care"
                elif "CMMC" in audit_framework:
                    system_role = "CMMC 2.0 Auditor. Focus on NIST 800-171 security controls and CUI data protection."
                    search_query = "Access control encryption CUI NIST 800-171"
                else:
                    system_role = "Hostile Medical AI Auditor. Focus on Article 10 (Data) and Article 14 (Human Oversight)."
                    search_query = "Article 10 Article 14 mandatory technical requirements"

                # Step 4: Run the AI
                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])
                
                prompt = f"""
                SYSTEM: You are a {system_role}. 
                Your job is to strictly verify if the USER EVIDENCE matches the LAW.
                
                LAW REFERENCE: {reg_context}
                USER EVIDENCE: {user_text}
                
                OUTPUT: 
                1. Provide a SCORE (0-10) for overall compliance.
                2. If it's a financial plan for Colorado, acknowledge the AI holdings as transparency proof.
                3. List exactly what is MISSING.
                """
                
                llm = get_llm()
                result = llm.invoke(prompt).content
                status.update(label="✅ Analysis Complete!", state="complete")
                
                # Step 5: Display Findings
                st.markdown(f"### 🏆 {audit_framework} AUDIT REPORT")
                st.info(result)
            else:
                st.error(f"Error: No PDF files found in '{selected_reg_path}'. Double-check your GitHub folder names!")
