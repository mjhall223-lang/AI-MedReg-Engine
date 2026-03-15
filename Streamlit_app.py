import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Remediation Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

# Construction order matters to prevent UI vanishing
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown(f"**Specialist:** Myia Hall")
    st.markdown("---")

    # DB STATUS with error handling
    try:
        db, files_found = load_multi_knowledge_base()
        if files_found:
            st.success(f"📚 {len(files_found)} PDFs Indexed")
            with st.expander("Manifest"):
                for f in files_found: st.text(f"📄 {f}")
        else:
            st.warning("⚠️ No PDFs found in 'Regulations/'")
    except:
        st.error("Engine path error.")
        db, files_found = None, []

    st.markdown("---")
    st.markdown("### 📈 FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (M):", value=50.0)
    est_replaced = st.number_input("Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Tax Liability", f"${impact['total']:,}")

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file or not db:
        st.error("Database missing or no upload found.")
    else:
        with st.status("🔍 CROSS-REFERENCING...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in chunks[:10]])
            
            # Fetch Context
            search_query = "EU IVDR Article 10, Article 14, SB24-205 reasonable care"
            search_docs = db.similarity_search(search_query, k=8)
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX: {impact['total']}
            
            REPORT:
            PART 1: AUDIT (Score 1-10. 1-4 is safe/compliant, 8-10 is high-risk).
            PART 2: REMEDIATION (Mandatory Worker Clause if score > 5).
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 AUDIT"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ STRATEGY"); st.markdown(report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            os.remove(tmp_path)
