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
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", value=50.0) # Users enter in millions
    est_replaced = st.number_input("Est. Human Roles Replaced:", value=50)
    
    # Calculate Impact using millions multiplier
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.warning("Please upload a document for evidence.")
    else:
        with st.status("🔍 ANALYZING COMPLIANCE & ECONOMIC ALIGNMENT...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # Load Policy Context
            vector_db = load_multi_knowledge_base(selected_frameworks)
            
            # Load and Chunk Evidence
            loader = PyPDFLoader(tmp_path)
            evidence_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in evidence_chunks[:10]])
            
            # Search for relevant policy snippets
            search_query = "EU AI Act High-Risk, Colorado SB24-205 duty of care, bias testing requirements"
            search_docs = vector_db.similarity_search(search_query, k=8) if vector_db else []
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            # THE REFINED PROMPT
            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            Compare the 'EVIDENCE' against the 'REGULATORY CONTEXT'.

            REGULATORY CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX LIABILITY: {impact['total']}

            MANDATORY REPORT STRUCTURE:
            PART 1: LEGAL AUDIT
            - Identify compliance with EU Law (IVDR/AI Act) and Colorado SB24-205.
            - Assess 'Reasonable Care'. 
            - Score 'Great Divergence' (1-10). 
              * Score 1-4: Project shows evidence of human oversight and risk management (Safe).
              * Score 8-10: Project shows total automation without oversight (Hazardous).

            PART 2: STRATEGIC REMEDIATION
            - If score < 5: DO NOT mandate legal clauses. Recommend 'Abundance Bonus' as a voluntary strategic move.
            - If score > 5: Draft a mandatory 'Worker Transition Clause'.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            # Split display
            col1, col2 = st.columns(2)
            with col1:
                st.error("### 📜 LEGAL AUDIT")
                st.markdown(report.split("PART 2:")[0])
            with col2:
                st.success("### 🛠️ STRATEGIC REMEDIATION")
                st.markdown("REMEDIATION" + report.split("PART 2:")[1] if "PART 2:" in report else "Audit passed. System compliant.")
            
            st.download_button("📄 Download PDF Report", create_pdf(report), file_name="Audit_Report.pdf")
            os.remove(tmp_path)
