import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. APP CONFIG ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** MJ Hall")
    
    audit_framework = st.selectbox(
        "Select Regulatory Framework",
        ["EU AI Act (Medical & IVDR)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. GITHUB FOLDER MAPPING ---
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 / NIST 800-171": "Regulations/Regulations/CMMC"
}
selected_reg_path = framework_folders[audit_framework]

# --- 4. ENGINE FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200).split_documents(loader.load()))
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")) if all_chunks else None

# --- 5. THE WORKFLOW ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")
st.markdown("---")

# This creates the visual "slot" for the report
results_container = st.container()

if st.button("🚀 Run Strict Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 Running Audit Engine...") as status:
            # Step A: Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            
            # Step B: Load the Law
            vector_db = load_knowledge_base(selected_reg_path)
            
            if vector_db:
                # Step C: Configure Brain
                is_premium = service_tier == "Premium Remediation"
                if "Colorado" in audit_framework:
                    role = "Colorado AI Act Lead. NOTE: AI stock holdings (NVDA, PLTR) are VALID transparency evidence."
                    query = "Algorithmic discrimination bias impact assessment"
                elif "CMMC" in audit_framework:
                    role = "CMMC 2.0 Auditor. Focus on CUI data protection."
                    query = "Access control encryption NIST 800-171"
                else:
                    role = "Strict Medical AI Auditor. Focus on Article 10/14 and IVDR Annex II."
                    query = "Article 10 Data Article 14 Oversight IVDR"

                # Step D: Retrieve and Generate
                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(query, k=5)])
                remedy = "Provide specific policy language to fix gaps." if is_premium else "List missing items only."
                
                prompt = f"SYSTEM: {role}\nLAW: {reg_context}\nEVIDENCE: {user_text}\nTASK: Provide a SCORE (0-10), list specific GAPS, and {remedy}"
                
                final_report = get_llm().invoke(prompt).content
                status.update(label="✅ Analysis Complete!", state="complete")
                
                # --- STEP 6: THE FORCE-DISPLAY ---
                with results_container:
                    st.success(f"### 📊 {audit_framework} REPORT")
                    st.markdown(final_report)
                    st.download_button("📩 Download Report", final_report, file_name="Audit_Report.md")
            else:
                st.error(f"Error: No PDFs found in {selected_reg_path}. Check GitHub!")
