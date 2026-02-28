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
        ["EU AI Act (Medical & IVDR)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    service_tier = st.radio("Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. DYNAMIC MAPPING ---
# These match your GitHub folders exactly based on your screenshots
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  # Looks in main folder for EU_regulations.pdf & lvdr.pdf
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 / NIST 800-171": "Regulations/Regulations/CMMC"
}
selected_reg_path = framework_folders[audit_framework]

# --- 4. CORE FUNCTIONS ---
@st.cache_resource
def get_llm():
    # Uses your GROQ_API_KEY from Streamlit Secrets
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
# The button is always visible
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
                # Step 3: Setup Auditor Personality
                if "Colorado" in audit_framework:
                    system_role = "Colorado AI Act Auditor. AI stock holdings (NVDA, PLTR) are VALID evidence for business transparency."
                    search_query = "Algorithmic discrimination bias impact assessment"
                elif "CMMC" in audit_framework:
                    system_role = "CMMC 2.0 Auditor. Focus on NIST 800-171 and CUI security."
                    search_query = "Access control encryption CUI protocols"
                else:
                    system_role = "Medical AI & IVDR Auditor. Focus on Article 10/14 and IVDR Annex II technical documentation."
                    search_query = "Article 10 Data Article 14 Oversight IVDR requirements"

                # Step 4: Run the AI
                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])
                
                prompt = f"""
                SYSTEM: You are a {system_role}. 
                Verify the USER EVIDENCE against the LAW provided.
                
                LAW REFERENCE: {reg_context}
                USER EVIDENCE: {user_text}
                
                OUTPUT: 
                1. Provide a SCORE (0-10) for overall compliance.
                2. Be detailed. If it's Colorado, mention if their AI stock picks count as transparency.
                3. List MISSING items in bullet points.
                """
                
                llm = get_llm()
                result = llm.invoke(prompt).content
                status.update(label="✅ Analysis Complete!", state="complete")
                
                # --- STEP 6: THE OUTPUT (The part that was missing!) ---
                st.markdown("---")
                st.success("### 📊 FINAL AUDIT REPORT")
                st.markdown(result) # This prints the AI's findings
                
                # Add a download button for the client
                st.download_button("📩 Download Audit Report", result, file_name="MJ_Hall_Audit.md")
            else:
                st.error(f"Error: No PDF files found in '{selected_reg_path}'. Check your GitHub folder names!")
