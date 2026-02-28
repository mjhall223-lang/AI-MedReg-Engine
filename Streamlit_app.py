import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. CONFIG ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** MJ Hall")
    audit_framework = st.selectbox("Framework", ["EU AI Act (Medical & IVDR)", "Colorado AI Act", "CMMC 2.0"])
    
    # This matches your screenshot
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])

# --- 2. GITHUB MAPPING ---
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0": "Regulations/Regulations/CMMC"
}
selected_reg_path = framework_folders[audit_framework]

# --- 3. FUNCTIONS ---
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

# --- 4. THE AUDIT ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")
results_container = st.container()

if st.button("🚀 Run Full Regulatory Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 CROSS-REFERENCING FEDERAL & STATE LAW...") as status:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            vector_db = load_knowledge_base(selected_reg_path)
            
            if vector_db:
                is_premium = service_tier == "Premium Remediation"
                
                # Set specific queries based on the framework
                if "Colorado" in audit_framework:
                    role = "Colorado Attorney General Enforcement Officer"
                    query = "Duty of care risk management algorithmic discrimination impact assessment"
                elif "CMMC" in audit_framework:
                    role = "DoD Cyber Auditor"
                    query = "NIST 800-171 security controls CUI"
                else:
                    role = "EU Notified Body Compliance Lead"
                    query = "Article 10 Data Article 14 Human Oversight"

                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(query, k=5)])
                
                # THE "GOAL-ORIENTED" PROMPT
                remediation_instruction = "Provide a full REMEDIATION PLAN with drafted policy language." if is_premium else "List missing requirements only."
                
                prompt = f"""
                SYSTEM: You are the {role}. You are strict and legally focused.
                LAW: {reg_context}
                EVIDENCE: {user_text}

                TASK:
                1. Provide a STATUS: [PASS] or [FAIL]. (Anything missing core legal policies is a FAIL).
                2. COMPLIANCE SCORE: [0-10].
                3. GAPS: Explain exactly which laws are being violated.
                4. {remediation_instruction}
                """
                
                final_report = get_llm().invoke(prompt).content
                status.update(label="✅ Analysis Complete!", state="complete")
                
                with results_container:
                    st.success("### 📜 OFFICIAL REGULATORY REPORT")
                    st.markdown(final_report)
                    st.download_button("📩 Export Audit", final_report, file_name="Regulatory_Audit.md")
            else:
                st.error("Could not find regulation database.")
