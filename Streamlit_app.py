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
    
    # DB STATUS CHECKER (Simplified: Just looks at the Regulations folder)
    db, files_found = load_multi_knowledge_base()
    
    if files_found:
        st.success(f"📚 Database Active: {len(files_found)} files indexed.")
        with st.expander("View Active Manifest"):
            for f in files_found: st.text(f"📄 {f}")
    else:
        st.error("⚠️ Database Empty! No PDFs found in the 'Regulations' folder.")
        st.info("💡 Tip: Upload PDFs to your GitHub repository in a folder named 'Regulations'.")

    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", value=50.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", value=50)
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file or not db:
        st.warning("Database is empty or no file uploaded.")
    else:
        with st.status("🔍 AUDITING...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
            user_text = "\n\n".join([c.page_content for c in chunks[:10]])
            
            # Fetch Context from Database
            search_query = "EU IVDR requirements, human oversight, algorithmic bias"
            search_docs = db.similarity_search(search_query, k=8)
            reg_context = "\n\n".join([f"(Doc: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])

            prompt = f"""
            SYSTEM: Senior Regulatory Architect. 
            Compare EVIDENCE against REGULATORY CONTEXT.
            REGULATORY CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            TAX LIABILITY: {impact['total']}
            
            REPORT:
            PART 1: LEGAL AUDIT (List Gaps and Divergence Score).
            PART 2: STRATEGIC REMEDIATION (Mandatory Transition Clause if score > 5).
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            
            col1, col2 = st.columns(2)
            with col1: st.error("### 📜 LEGAL AUDIT"); st.markdown(report.split("PART 2:")[0])
            with col2: st.success("### 🛠️ STRATEGIC REMEDIATION"); st.markdown(report.split("PART 2:")[1] if "PART 2:" in report else "Compliant.")
            os.remove(tmp_path)
