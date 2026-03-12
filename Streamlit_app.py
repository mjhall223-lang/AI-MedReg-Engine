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
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=50000.0)
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
        with st.status("🔍 GATHERING POLICY CONTEXT & DRAFTING REMEDIATION...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            
            vector_db = load_multi_knowledge_base(selected_frameworks, selected_fed_docs)
            st.session_state.vector_db = vector_db
            
            search_query = "EU AI Act High-Risk, Colorado AI Act Section 3, Robot Tax, Great Divergence, Abundance Bonus"
            search_docs = vector_db.similarity_search(search_query, k=25) if vector_db else []
            reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            
            prompt = f"""
            SYSTEM: Senior Regulatory Architect & Economic Analyst.
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            ECONOMIC ESTIMATE: {impact}

            MANDATORY REPORTING STRUCTURE:
            
            PART 1: AUDIT FINDINGS
            - Analyze EU AI Act / Colorado AI Act High-Risk classifications in detail.
            - Evaluate 'Robot Tax' liability referencing the WH and CPA docs.
            - Score the 'Great Divergence' (1-10) with reasoning based on the $3M savings vs 50 roles replaced.

            PART 2: REMEDIATION
            - You MUST provide this section if roles are replaced or score is < 9.
            - DRAFT FULL LEGAL ARTICLES for a 'Worker Transition & Dividend Clause'.
            - Include: 5% Abundance Bonuses, Re-skilling fund allocations, and TX_CPA-compliant logging.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1:
                st.error("### 📜 AUDIT FINDINGS")
                # Ensure we capture everything before the remediation starts
                st.markdown(report.split("PART 2:")[0] if "PART 2:" in report else report)
            with col2:
                st.success("### 🛠️ PROPOSED REMEDIATION")
                if "PART 2:" in report:
                    st.markdown("### REMEDIATION STRATEGY" + report.split("PART 2:")[1])
                else:
                    st.info("System efficiency meets abundance standards. No major remediation required.")
            
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
