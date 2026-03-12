import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    selected_frameworks = st.multiselect("Active Frameworks", ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0"], default=["Federal Proposal", "EU AI Act"])
    st.markdown("---"); st.markdown("### 📜 FEDERAL KNOWLEDGE BASE")
    fed_files = [f for root, dirs, files in os.walk("Regulations") if "Federal" in root for f in files if f.endswith(".pdf")]
    selected_fed_docs = st.multiselect("Active Policy Docs", options=fed_files, default=fed_files)
    st.markdown("---"); st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=50.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", min_value=0, value=50)
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")
    if st.button("🗑️ Reset Engine"):
        st.session_state.clear(); st.rerun()

uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit & Remediation"):
    if not uploaded_file:
        st.warning("Please upload a document!")
    else:
        with st.status("🔍 ANALYZING RISK & ENFORCING REMEDIATION...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            
            vector_db = load_multi_knowledge_base(selected_frameworks, selected_fed_docs)
            st.session_state.vector_db = vector_db
            
            search_query = "AI tax, labor displacement, worker transition plan, abundance bonus, Section 1701"
            search_docs = vector_db.similarity_search(search_query, k=25) if vector_db else []
            reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            
            prompt = f"""
            SYSTEM: Senior Regulatory Architect & Economic Analyst.
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            ECONOMIC ESTIMATE: {impact}

            INSTRUCTIONS:
            1. Audit for EU AI Act / Colorado AI Act (High-Risk classifications).
            2. Evaluate 'Robot Tax' liability based on Federal Proposal docs.
            3. Score the 'Great Divergence' (1-10). 
            
            REMEDIATION (PREMIUM):
            - IF Human Roles Replaced > 0 OR 'Great Divergence' score < 9, you MUST provide a 'REMEDIATION' section.
            - DRAFT THE FULL TEXT of a 'Worker Transition & Dividend Clause'.
            - DO NOT just say remediation is required; ACTUALLY WRITE THE CLAUSES.
            - Include specific legally-structured language for: 
                a) 5% Abundance Bonuses from cost-savings.
                b) Re-skilling / Upskilling fund allocations.
                c) TX_CPA-compliant audit logging protocols.
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
                    st.markdown("REMEDIATION" + report.split("REMEDIATION")[1])
                else:
                    st.info("System meets abundance standards. No major remediation required.")
            
            st.download_button("📄 Download Certified Report", create_pdf(report), file_name="Certified_Audit_Report.pdf")
            os.remove(tmp_path)

if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Ask a follow-up..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            ctx = ""
            if st.session_state.vector_db:
                docs = st.session_state.vector_db.similarity_search(user_input, k=15)
                ctx = "\n\n".join([d.page_content for d in docs])
            st.markdown(get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {ctx}\nUSER: {user_input}").content)
