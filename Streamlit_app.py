import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- UI CONFIG ---
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: 
    st.session_state.messages = []

# --- SIDEBAR: AUDIT CONTROLS ---
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

# --- MAIN ENGINE EXECUTION ---
uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit & Remediation"):
    if not uploaded_file:
        st.warning("Please upload a document!")
    else:
        with st.status("🔍 GATHERING POLICY CONTEXT & DRAFTING REMEDIATION...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            # 1. LOAD THE KNOWLEDGE BASE (REGULATIONS)
            vector_db = load_multi_knowledge_base(selected_frameworks)
            st.session_state.vector_db = vector_db
            
            # 2. EXTRACT & CHUNK THE EVIDENCE (THE UPLOADED PDF)
            # This was the missing part! We now load the evidence into a searchable context.
            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            evidence_chunks = text_splitter.split_documents(raw_docs)
            
            # We take the most substantial parts of your evidence (first ~10k tokens)
            user_evidence_text = "\n\n".join([c.page_content for c in evidence_chunks[:10]])
            
            # 3. SEARCH THE REGULATIONS FOR RELEVANCE
            search_query = "EU AI Act High-Risk, Colorado SB-205, Great Divergence metrics, Robot Tax liability"
            search_docs = vector_db.similarity_search(search_query, k=8) if vector_db else []
            reg_context = "\n\n".join([f"(Source: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            
            # 4. THE MASTER PROMPT (Evidence + Context)
            prompt = f"""
            SYSTEM: Senior Regulatory Architect & AI Economic Analyst.
            
            INSTRUCTIONS:
            You are auditing a project based on the 'EVIDENCE' provided. 
            Compare the evidence against the 'REGULATORY CONTEXT'.
            
            REGULATORY CONTEXT:
            {reg_context}

            EVIDENCE TO AUDIT:
            {user_evidence_text}

            ECONOMIC DATA:
            Estimated Monthly Tokens: {est_tokens}M
            Roles Replaced: {est_replaced}
            Calculated Liability: {impact['total']}

            MANDATORY REPORT STRUCTURE:
            PART 1: AUDIT FINDINGS
            - Classify the 'Evidence' under EU/Colorado risk tiers.
            - Explain the 'Robot Tax' liability using the Economic Data.
            - Assign a 'Great Divergence' score (1-10) for this implementation.

            PART 2: REMEDIATION
            - DRAFT A LEGAL 'WORKER TRANSITION & DIVIDEND CLAUSE'.
            - Include 5% Abundance Bonuses and re-skilling fund allocations.
            """
            
            try:
                # 5. GENERATE THE REPORT
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.final_report = report
                
                # UI Display
                col1, col2 = st.columns(2)
                with col1:
                    st.error("### 📜 AUDIT FINDINGS")
                    st.markdown(report.split("PART 2:")[0] if "PART 2:" in report else report)
                with col2:
                    st.success("### 🛠️ PROPOSED REMEDIATION")
                    if "PART 2:" in report:
                        st.markdown("### STRATEGY" + report.split("PART 2:")[1])
                
                st.download_button("📄 Download Certified Report", create_pdf(report), file_name="Audit_Report.pdf")
            
            except Exception as e:
                st.error(f"Engine Error: {str(e)}")
            finally:
                os.remove(tmp_path)

# --- FOLLOW-UP CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Ask a follow-up about the audit..."):
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            ctx = ""
            if st.session_state.vector_db:
                docs = st.session_state.vector_db.similarity_search(user_input, k=5)
                ctx = "\n\n".join([d.page_content for d in docs])
            st.markdown(get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {ctx}\nUSER: {user_input}").content)
