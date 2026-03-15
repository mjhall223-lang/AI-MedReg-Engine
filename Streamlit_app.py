import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

# 1. SIDEBAR: Keep this simple so it doesn't crash
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown(f"**Specialist:** Myia Hall")
    st.markdown("---")

    # Only load DB when requested to prevent sidebar vanishing
    if st.button("🔄 Sync Knowledge Base"):
        with st.spinner("Crawling folders..."):
            db, files = load_multi_knowledge_base()
            if files:
                st.session_state.db = db
                st.session_state.files = files
                st.success(f"Indexed {len(files)} PDFs")
            else:
                st.error("No PDFs found in 'Regulations/'")

    if "files" in st.session_state:
        with st.expander("Active Manifest"):
            for f in st.session_state.files: st.text(f"📄 {f}")

    st.markdown("---")
    st.markdown("### 📈 FORECASTER")
    tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    replaced = st.number_input("Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=tokens*1000000, replaced_staff=replaced)
    st.metric("Tax Liability", f"${impact['total']:,}")

# 2. MAIN BODY
uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.error("Please upload a file first.")
    elif "db" not in st.session_state:
        st.error("Please click 'Sync Knowledge Base' in the sidebar first!")
    else:
        with st.status("🔍 CROSS-REFERENCING...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in chunks[:10]])
            
            # Fetch Context
            search_docs = st.session_state.db.similarity_search("AI high risk, oversight, bias", k=8)
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX: {impact['total']}
            
            PART 1: AUDIT (1-10 Score. 1-4 is safe, 8-10 is high-risk).
            PART 2: REMEDIATION (Worker Clause if score > 5).
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 AUDIT"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ STRATEGY"); st.markdown(report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            os.remove(tmp_path)
            
