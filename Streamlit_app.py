import streamlit as st
import os
import tempfile
from engine import get_llm, load_selected_docs, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

# --- SIDEBAR: DYNAMIC FILE TOGGLES ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown(f"**Specialist:** Myia Hall")
    st.markdown("---")

    # FIND ALL FILES ONCE
    all_pdfs = []
    for root, dirs, files in os.walk("Regulations"):
        for file in files:
            if file.endswith(".pdf"): all_pdfs.append(file)

    if all_pdfs:
        st.markdown("### 📜 ACTIVE KNOWLEDGE BASE")
        selected_files = []
        for f in sorted(all_pdfs):
            if st.checkbox(f"📄 {f}", value=True):
                selected_files.append(f)
        st.info(f"Indexing {len(selected_files)} files.")
    else:
        st.error("No PDFs found in 'Regulations/'")
        selected_files = []

    st.markdown("---")
    st.markdown("### 📈 FORECASTER")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=tokens*1000000, replaced_staff=replaced)
    st.metric("Tax Liability", f"${impact['total']:,}")

# --- MAIN BODY: GAP ANALYSIS & REMEDIATION ---
uploaded_file = st.file_uploader("Upload Project Evidence", type="pdf")

if st.button("🚀 Run Gap Analysis & Remediation Pathway"):
    if not uploaded_file or not selected_files:
        st.error("Missing Evidence or Knowledge Base Toggles.")
    else:
        with st.status("🔍 PERFORMING GAP ANALYSIS...") as status:
            # 1. Sync Knowledge Base from Toggles
            db = load_selected_docs(selected_files)
            
            # 2. Process Evidence
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            evidence_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            evidence_text = "\n\n".join([c.page_content for c in evidence_chunks[:10]])
            
            # 3. Retrieve Context
            search_docs = db.similarity_search("High risk AI, human oversight, algorithmic bias", k=8)
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            # 4. Prompt for Gap Analysis
            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            Perform a GAP ANALYSIS between EVIDENCE and SELECTED REGS.
            
            CONTEXT: {reg_context}
            EVIDENCE: {evidence_text}
            TAX LIABILITY: {impact['total']}
            
            REPORT STRUCTURE:
            PART 1: GAP ANALYSIS
            - Identify exactly what is missing from the Evidence based on the active Regs.
            - Score 'Great Divergence' (1-10).
            
            PART 2: REMEDIATION PATHWAY
            - Provide a step-by-step pathway to fix the identified Gaps.
            - Include 'Worker Transition' mandates if Score > 5.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 GAP ANALYSIS"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ REMEDIATION PATHWAY"); st.markdown(report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            os.remove(tmp_path)

if "final_report" in st.session_state:
    st.download_button("📄 Download Audit", create_pdf(st.session_state.final_report), file_name="Gap_Analysis.pdf")
