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
    pdf.cell(0, 10, "OFFICIAL REGULATORY AUDIT FINDING", ln=True, align='C')
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
                    is_premium = service_tier == "Premium Remediation"
                    
                    if "Colorado" in audit_framework:
                        role, query = "Colorado Attorney General Enforcement Officer", "Duty of care risk management algorithmic discrimination"
                    elif "CMMC" in audit_framework:
                        role, query = "DoD Cyber Auditor", "NIST 800-171 security controls CUI"
                    elif "Bias" in audit_framework:
                        role, query = "Clinical Equity & Bias Auditor", "Article 10 bias mitigation demographic representation"
                    elif "FDA" in audit_framework:
                        role, query = "FDA Digital Health Specialist", "PCCP Modification Protocol Impact Assessment"
                    else:
                        role, query = "EU Notified Body Compliance Lead", "Article 10 Data Quality Article 14 Human Oversight"

                    docs = vector_db.similarity_search(query, k=5)
                    context_list = []
                    for d in docs:
                        source_name = os.path.basename(d.metadata.get('source', 'Regulation'))
                        page_num = d.metadata.get('page', 'N/A')
                        context_list.append(f"SOURCE: {source_name} (Page {page_num}):\n{d.page_content}")
                    
                    reg_context = "\n\n----- \n\n".join(context_list)
                    
                    # THE "STRICT" PROMPT INSTRUCTIONS
                    prompt = f"""
                    SYSTEM: You are the {role}. You are strictly legally focused and objective. 
                    STRICTNESS RULE: Do not infer or assume policies exist. If a requirement is not explicitly stated in the EVIDENCE, it does not exist. 

                    LAW (REFERENCE): 
                    {reg_context}
                    
                    EVIDENCE (UPLOADED): 
                    {user_text}

                    TASK:
                    1. STATUS: Provide one: [PASS], [MINOR NON-CONFORMANCE], or [MAJOR NON-CONFORMANCE/FAIL].
                       - Any missing core legal anchor (like an Impact Assessment, Data Governance policy, or Acceptance Criteria) MUST be a [MAJOR NON-CONFORMANCE/FAIL].
                    2. COMPLIANCE SCORE: [0-10]. Be brutal.
                    3. GAPS: Cite specific laws, Articles, or Guidance Page Numbers.
                    4. RISK ASSESSMENT: For every gap, explain the regulatory risk (e.g., FDA Warning Letter, legal liability, or market withdrawal).
                    5. {"Provide a full REMEDIATION PLAN." if is_premium else "List missing requirements only."}
                    """
                    
                    final_report = get_llm().invoke(prompt).content
                    status.update(label="✅ Audit Complete!", state="complete")
                    
                    with results_container:
                        st.error("### 📜 OFFICIAL REGULATORY FINDINGS")
                        st.markdown(final_report)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button("📩 Export Markdown", final_report, file_name="Audit_Finding.md")
                        with col2:
                            pdf_data = create_pdf(final_report)
                            st.download_button("📄 Export Official PDF", pdf_data, file_name="Official_Audit.pdf", mime="application/pdf")
                else:
                    st.error(f"KB Error: {selected_reg_path}")

            except Exception as e:
                st.error(f"Processing Error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
