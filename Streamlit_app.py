import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fpdf import FPDF

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    audit_framework = st.selectbox("Framework", [
        "EU AI Act (Medical & IVDR)", 
        "Colorado AI Act", 
        "CMMC 2.0",
        "Medical Bias & Health Equity (Article 10)",
        "FDA PCCP (Clinical Change)"
    ])
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])

# --- 2. GITHUB MAPPING ---
# Matches your exact path: Regulations/Regulations/Federal
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0": "Regulations/Regulations/CMMC",
    "Medical Bias & Health Equity (Article 10)": ".",
    "FDA PCCP (Clinical Change)": "Regulations/Regulations/Federal"
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

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL REGULATORY AUDIT FINDING", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Lead Specialist: Myia Hall | Framework: {audit_framework}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())

# --- 4. THE AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

if st.button("🚀 Run Full Regulatory Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 ANALYZING AGAINST INTERNAL & FEDERAL STANDARDS...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                vector_db = load_knowledge_base(selected_reg_path)
                
                if vector_db:
                    st.session_state.vector_db = vector_db
                    
                    # Persona Mapping
                    role_map = {
                        "Colorado AI Act": "Colorado Attorney General",
                        "CMMC 2.0": "DoD Cyber Auditor",
                        "Medical Bias & Health Equity (Article 10)": "Clinical Equity Auditor",
                        "FDA PCCP (Clinical Change)": "FDA Digital Health Specialist",
                        "EU AI Act (Medical & IVDR)": "EU Notified Body Lead"
                    }
                    role = role_map.get(audit_framework, "Regulatory Lead")
                    st.session_state.current_role = role

                    # Retrieve context (including your new Doc of Docs!)
                    docs = vector_db.similarity_search("Mandatory requirements and missing document gaps", k=6)
                    reg_context = "\n\n".join([f"SOURCE: {os.path.basename(d.metadata['source'])}:\n{d.page_content}" for d in docs])
                    
                    prompt = f"""
                    SYSTEM: You are the {role}. You are strictly focused on identifying non-conformance.
                    INTERNAL STANDARDS & LAW: 
                    {reg_context}
                    
                    EVIDENCE: 
                    {user_text}

                    TASK:
                    1. STATUS: [PASS], [MINOR NON-CONFORMANCE], or [MAJOR NON-CONFORMANCE/FAIL].
                    2. SCORING RULE: If any core pillar (AMP, FRIA, SSP, or Impact Assessment) from the INTERNAL STANDARDS is missing, the score MUST be below 5/10.
                    3. GAPS: Cite specific requirements from the INTERNAL STANDARDS docs.
                    4. {"REMEDIATION: Provide draft policy language." if service_tier == "Premium Remediation" else "List missing requirements."}
                    """
                    
                    report = get_llm().invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Audit Complete!", state="complete")
                    
                    st.error("### 📜 OFFICIAL AUDIT FINDINGS")
                    st.markdown(report)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("📩 Export Markdown", report, file_name="Audit.md")
                    with col2:
                        st.download_button("📄 Export Official PDF", create_pdf(report), file_name="Official_Audit.pdf", mime="application/pdf")
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- 5. INTERACTIVE REMEDIATION CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Interactive Auditor Chat")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_input := st.chat_input("Ex: How do I fix the AMP threshold gap?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=3)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            
            response = get_llm().invoke(f"ROLE: {st.session_state.current_role}\nCONTEXT: {context_text}\nQUESTION: {user_input}\nUse context to explain remediation steps.").content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
