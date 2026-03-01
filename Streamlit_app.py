import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fpdf import FPDF

# --- 1. CONFIG ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# Initialize Chat History for the Auditor
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** MJ Hall")
    
    audit_framework = st.selectbox("Framework", [
        "EU AI Act (Medical & IVDR)", 
        "Colorado AI Act", 
        "CMMC 2.0",
        "Medical Bias & Health Equity (Article 10)",
        "FDA PCCP (Clinical Change)"
    ])
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 2. GITHUB MAPPING ---
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
    if not all_chunks:
        return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL REGULATORY FINDING", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, "Auditor: MJ Hall | Generated via AI-MedReg-Engine", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())

# --- 4. THE AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")
results_container = st.container()

if st.button("🚀 Run Full Regulatory Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 CROSS-REFERENCING FEDERAL & STATE LAW...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                vector_db = load_knowledge_base(selected_reg_path)
                
                if vector_db:
                    st.session_state.vector_db = vector_db # Save for chat
                    
                    if "Colorado" in audit_framework:
                        role, query = "Colorado Attorney General", "Duty of care risk management discrimination"
                    elif "CMMC" in audit_framework:
                        role, query = "DoD Cyber Auditor", "NIST 800-171 System Security Plan SSP"
                    elif "Bias" in audit_framework:
                        role, query = "Clinical Equity Auditor", "Article 10 bias mitigation medical AI"
                    elif "FDA" in audit_framework:
                        role, query = "FDA Digital Health Specialist", "PCCP Algorithm Modification Protocol"
                    else:
                        role, query = "EU Compliance Lead", "Article 10 Data Governance Article 14 Oversight"

                    st.session_state.current_role = role
                    
                    docs = vector_db.similarity_search(query, k=5)
                    reg_context = "\n\n".join([f"SOURCE: {os.path.basename(d.metadata['source'])} (Page {d.metadata.get('page','N/A')}):\n{d.page_content}" for d in docs])
                    
                    prompt = f"""
                    SYSTEM: You are the {role}. Strict Zero-Tolerance Auditor. 
                    REFERENCE LAW: {reg_context}
                    EVIDENCE: {user_text}

                    TASK:
                    1. STATUS: [PASS], [MINOR NON-CONFORMANCE], or [MAJOR NON-CONFORMANCE/FAIL].
                    2. SCORE: [0-10]. 
                    3. GAPS: Cite specific laws and page numbers.
                    4. RISK: Explain regulatory consequences.
                    5. {"REMEDIATION: Provide draft language." if service_tier == "Premium Remediation" else "List missing requirements."}
                    """
                    
                    report = get_llm().invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Audit Complete!", state="complete")
                    
                    with results_container:
                        st.error("### 📜 OFFICIAL REGULATORY FINDINGS")
                        st.markdown(report)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("📩 Export Markdown", report, file_name="Audit.md")
                        with col2:
                            pdf_data = create_pdf(report)
                            st.download_button("📄 Export Official PDF", pdf_data, file_name="Official_Audit.pdf", mime="application/pdf")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- 5. INTERACTIVE REMEDIATION CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Interactive Auditor Chat")
    st.info("Ask the auditor how to fix the gaps identified above.")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ex: How do I draft a Data Governance policy?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            # Use RAG to answer from the knowledge base
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=3)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            
            chat_prompt = f"""
            ROLE: {st.session_state.current_role}
            CONTEXT: {context_text}
            QUESTION: {user_input}
            INSTRUCTION: Use the context to explain how to fix the gap. Cite the document names.
            """
            
            response = get_llm().invoke(chat_prompt).content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
