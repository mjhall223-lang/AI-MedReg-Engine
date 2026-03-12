import streamlit as st
import os
import tempfile
import datetime
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

# --- ENGINE.PY LOGIC ---

def get_llm(is_cloud, st_secrets):
    if is_cloud:
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            api_key=st_secrets["GROQ_API_KEY"]
        )
    return ChatOllama(model="gemma2:2b", temperature=0)

def load_multi_knowledge_base(selected_frameworks, selected_fed_docs=[], root_folder="Regulations"):
    all_chunks = []
    if not os.path.exists(root_folder):
        return None

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".pdf"):
                path_lower = root.lower()
                is_framework = any(f.split()[0].lower() in path_lower for f in selected_frameworks)
                is_toggled_doc = (file in selected_fed_docs)

                if is_framework or is_toggled_doc:
                    full_path = os.path.join(root, file)
                    try:
                        loader = PyPDFLoader(full_path)
                        docs = loader.load()
                        for d in docs:
                            d.metadata["source_file"] = file
                        
                        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
                        all_chunks.extend(splitter.split_documents(docs))
                    except:
                        continue
    
    if not all_chunks:
        return None
    
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

class EconomicImpact:
    @staticmethod
    def calculate_liability(token_usage=0, replaced_staff=0):
        # 2026 AI Dividend Logic: $0.0005 per 1k tokens + 15% Shadow Payroll Tax
        token_tax = (token_usage / 1000) * 0.0005
        payroll_tax = (replaced_staff * 60000) * 0.15
        return {
            "token_tax": round(token_tax, 2),
            "payroll_tax": round(payroll_tax, 2),
            "total": round(token_tax + payroll_tax, 2)
        }

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "READY-AUDIT: CERTIFIED REGULATORY REPORT", ln=True, align='C')
    pdf.set_font("Arial", size=11)
    
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, "OFFICIAL CERTIFICATION:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Date: {datetime.date.today()}", ln=True)
    pdf.cell(0, 10, "Lead Specialist: Myia Hall", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "X________________________________________", ln=True)
    pdf.cell(0, 10, "Signature of Regulatory Architect", ln=True)
    
    return bytes(pdf.output())

# --- STREAMLIT_APP.PY INTERFACE ---

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    selected_frameworks = st.multiselect(
        "Active Frameworks", 
        ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0"],
        default=["Federal Proposal", "EU AI Act"]
    )

    st.markdown("---")
    st.markdown("### 📜 FEDERAL KNOWLEDGE BASE")
    fed_files = []
    for root, dirs, files in os.walk("Regulations"):
        if "Federal" in root:
            for f in files:
                if f.endswith(".pdf"): fed_files.append(f)
    
    selected_fed_docs = st.multiselect("Active Policy Docs", options=fed_files, default=fed_files)

    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=50.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", min_value=0, value=50)
    
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

    if st.button("🗑️ Reset Engine"):
        st.session_state.clear()
        st.rerun()

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit & Remediation"):
    if not uploaded_file:
        st.warning("Please upload a document to audit!")
    else:
        with st.status("🔍 ANALYZING RISK & ENFORCING REMEDIATION...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            vector_db = load_multi_knowledge_base(selected_frameworks, selected_fed_docs)
            st.session_state.vector_db = vector_db
            
            search_query = "AI tax, labor displacement, worker transition plan, abundance bonus, Section 1701"
            search_docs = vector_db.similarity_search(search_query, k=25) if vector_db else []
            reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            
            # UPDATED LOGIC: Strict trigger for displacement
            prompt = f"""
            SYSTEM: Senior Regulatory Architect & Economic Analyst.
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            ECONOMIC ESTIMATE: {impact}

            INSTRUCTIONS:
            1. Audit for EU AI Act / Colorado AI Act (High-Risk classifications).
            2. Evaluate 'Robot Tax' liability based on Federal Proposal docs.
            3. Score the 'Great Divergence' (1-10). 
            
            STRICT REMEDIATION LOGIC:
            - If Human Roles Replaced > 0, you MUST provide 'REMEDIATION'.
            - If 'Great Divergence' score is BELOW 9, you MUST provide 'REMEDIATION'.
            - Draft a 'Worker Transition & Dividend Clause' including a 5% Abundance Bonus.
            - Ensure remediation references the TX_CPA_AI_Tax_Compliance audit trail.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1:
                st.error("### 📜 AUDIT FINDINGS")
                st.markdown(report.split("REMEDIATION")[0])
            with col2:
                st.success("### 🛠️ PROPOSED REMEDIATION")
                if "REMEDIATION" in report:
                    st.markdown(report.split("REMEDIATION")[1])
                else:
                    st.info("System efficiency meets abundance standards. No major remediation required.")
            
            st.download_button("📄 Download Certified Report", create_pdf(report), file_name="Certified_Audit_Report.pdf")
            os.remove(tmp_path)

# --- CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Ask a follow-up about the 'Productivity-Distribution Equation'..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            context_text = ""
            if st.session_state.vector_db:
                docs = st.session_state.vector_db.similarity_search(user_input, k=15)
                context_text = "\n\n".join([d.page_content for d in docs])
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
