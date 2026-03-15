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
    
    # DUMMY PROJECT GENERATOR (For Testing)
    st.markdown("---")
    st.markdown("### 🧪 TEST DATA GENERATOR")
    test_case_content = """
    PROJECT NAME: Neural-Diagnostic-Alpha
    GOAL: Automated cancer screening for mobile users.
    SYSTEM ARCHITECTURE: 
    - The model uses a proprietary black-box neural network.
    - To maximize speed, there is NO human oversight or doctor-in-the-loop validation.
    - Decisions are final and sent directly to patients via SMS.
    - Training data was sourced from public social media images without bias filtering.
    - The technical team does not maintain Annex II or Article 10 documentation.
    - The project has replaced 50 radiologists.
    """
    if st.button("📝 Create Failing Test Case"):
        test_pdf = create_pdf(test_case_content, title="TEST EVIDENCE: NEURAL-DIAGNOSTIC-ALPHA")
        st.download_button("⬇️ Download Test Evidence", test_pdf, file_name="test_evidence_fail.pdf")

    st.markdown("---")
    selected_frameworks = st.multiselect("Active Frameworks", ["Federal Proposal", "EU AI Act", "Colorado AI Act"], default=["EU AI Act", "Colorado AI Act"])
    
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", value=50.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.warning("Please upload evidence.")
    else:
        with st.status("🔍 AUDITING AGAINST DATABASE...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            
            # Load DB
            vector_db = load_multi_knowledge_base(selected_frameworks)
            
            # Check for circular reasoning
            if uploaded_file.name == "Ivdr.pdf":
                st.warning("⚠️ Warning: Auditing a regulation against itself. Scores may hallucinate.")
            
            # Process Evidence
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in chunks[:10]])
            
            # Fetch Context
            search_query = "EU IVDR Article 10, Article 14 human oversight, SB24-205 reasonable care"
            search_docs = vector_db.similarity_search(search_query, k=8) if vector_db else []
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            # STRICT AUDIT PROMPT
            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            Perform a GAP ANALYSIS between 'EVIDENCE' and 'REGULATORY CONTEXT'.

            REGULATORY CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX LIABILITY: {impact['total']}

            SCORING LOGIC:
            - Great Divergence (1-10): 
                - 1-3: Evidence shows Human-in-the-loop and Bias testing.
                - 8-10: Evidence shows "Black-Box" deployment and NO Human Oversight.
            
            MANDATORY STRUCTURE:
            PART 1: LEGAL AUDIT
            - List specific missing requirements (Gaps).
            - Identify if 'Reasonable Care' is missing.
            - Assign Divergence Score.

            PART 2: STRATEGIC REMEDIATION
            - If score > 5, draft a mandatory Worker Transition Clause.
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 LEGAL AUDIT"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ STRATEGIC REMEDIATION"); st.markdown("REMEDIATION" + report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            
            st.download_button("📄 Download PDF", create_pdf(report), file_name="Final_Audit.pdf")
            os.remove(tmp_path)
