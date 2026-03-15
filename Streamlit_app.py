import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# UI CONFIG
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: 
    st.session_state.messages = []

# SIDEBAR CONTROLS
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    selected_frameworks = st.multiselect(
        "Active Frameworks", 
        ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0"], 
        default=["Federal Proposal", "EU AI Act"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=50000.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", min_value=0, value=50)
    
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")
    
    if st.button("🗑️ Reset Engine"):
        st.session_state.clear()
        st.rerun()

# AUDIT EXECUTION
uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit & Remediation"):
    if not uploaded_file:
        st.warning("Please upload a document!")
    else:
        with st.status("🔍 GATHERING POLICY CONTEXT & DRAFTING REMEDIATION...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # 1. RAG Retrieval
            vector_db = load_multi_knowledge_base(selected_frameworks)
            st.session_state.vector_db = vector_db
            
            # 2. Chunk Evidence (Token Safety)
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            evidence_chunks = text_splitter.split_documents(raw_docs)
            user_text_summary = "\n\n".join([c.page_content for c in evidence_chunks[:12]])
            
            # 3. Policy Search
            search_query = "EU AI Act High-Risk, Colorado AI Act, Robot Tax Liability"
            search_docs = vector_db.similarity_search(search_query, k=10) if vector_db else []
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            
            # 4. Prompt Engineering
            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            CONTEXT: {reg_context}
            EVIDENCE: {user_text_summary}
            ECONOMIC IMPACT: {impact}

            MANDATORY REPORT:
            PART 1: AUDIT FINDINGS (High-Risk classifications, 'Robot Tax' score 1-10).
            PART 2: REMEDIATION (Draft legal articles for Worker Transition & Dividend Clause).
            """
            
            try:
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.final_report = report
                
                col1, col2 = st.columns(2)
                with col1:
                    st.error("### 📜 AUDIT FINDINGS")
                    st.markdown(report.split("PART 2:")[0] if "PART 2:" in report else report)
                with col2:
                    st.success("### 🛠️ PROPOSED REMEDIATION")
                    if "PART 2:" in report:
                        st.markdown("### REMEDIATION" + report.split("PART 2:")[1])
                
                st.download_button("📄 Download Report", create_pdf(report), file_name="Audit_Report.pdf")
            except Exception as e:
                st.error("Engine Timeout or Context Limit hit. Please try a smaller PDF.")
            finally:
                os.remove(tmp_path)

# CHAT INTERFACE
if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Follow-up on the audit..."):
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            ctx = ""
            if st.session_state.vector_db:
                docs = st.session_state.vector_db.similarity_search(user_input, k=5)
                ctx = "\n\n".join([d.page_content for d in docs])
            st.markdown(get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {ctx}\nUSER: {user_input}").content)
