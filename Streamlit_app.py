import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    selected_frameworks = st.multiselect("Active Frameworks", ["Federal Proposal", "EU AI Act", "Colorado AI Act"], default=["EU AI Act", "Colorado AI Act"])
    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", value=50000.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.warning("Upload a doc!")
    else:
        with st.status("🔍 AUDITING...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            
            vector_db = load_multi_knowledge_base(selected_frameworks)
            
            # Context & Evidence Loading
            loader = PyPDFLoader(tmp_path)
            evidence_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in evidence_chunks[:10]])
            
            search_query = "EU AI Act High-Risk, Colorado SB24-205 duty of care, algorithmic discrimination"
            search_docs = vector_db.similarity_search(search_query, k=8) if vector_db else []
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            # THE UPDATED PROMPT: Strict Legal Grounding
            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            Compare the 'EVIDENCE' against the 'REGULATORY CONTEXT'.

            REGULATORY CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX LIABILITY: {impact['total']}

            MANDATORY REPORT STRUCTURE:
            PART 1: LEGAL AUDIT
            - Classify under EU/Colorado Law. (Be precise: EU IVDR vs Colorado SB24-205).
            - Identify if 'Reasonable Care' is met. 
            - Score 'Great Divergence' (1-10). NOTE: If the project has human oversight and bias testing, the score MUST be low (2-4). High scores (8-10) are ONLY for unregulated automation.

            PART 2: STRATEGIC REMEDIATION
            - If score > 5: Draft a Worker Transition Clause.
            - If score < 5: Recommend 'Abundance Bonus' as a voluntary strategic move, NOT a legal requirement.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 LEGAL AUDIT"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ STRATEGIC REMEDIATION"); st.markdown("REMEDIATION" + report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            
            st.download_button("📄 Download Report", create_pdf(report), file_name="Audit.pdf")
            os.remove(tmp_path)
            
