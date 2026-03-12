import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework & Economic Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    selected_frameworks = st.multiselect("Active Frameworks", ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0"], default=["Federal Proposal", "EU AI Act"])
    
    st.markdown("---")
    st.markdown("### 📈 ECONOMIC FORECASTER")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=10.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", min_value=0, value=1)
    
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    st.metric("Estimated Tax Liability", f"${impact['total']:,}")
    
    if st.button("🗑️ Reset Engine"):
        st.session_state.clear()
        st.rerun()

uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.warning("Upload a document first!")
    else:
        with st.status("🔍 ANALYZING REGULATORY & ECONOMIC RISK...") as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            vector_db = load_multi_knowledge_base(selected_frameworks)
            st.session_state.vector_db = vector_db
            
            search_docs = vector_db.similarity_search("AI tax, labor displacement, productivity pay gap, Section 1701", k=25)
            reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
            user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
            
            prompt = f"""
            SYSTEM: Senior Regulatory Architect & Economic Analyst.
            CONTEXT: {reg_context}
            EVIDENCE: {user_text}
            ESTIMATED LIABIITY: {impact}

            INSTRUCTIONS:
            1. Audit for EU AI Act and Colorado SB 24-205 compliance.
            2. Evaluate potential 'Robot Tax' liability based on Federal Proposal documents.
            3. Score the 'Great Divergence' impact: Is this system hoarding productivity gains or sharing them?

            OUTPUT:
            - STATUS: (Pass/Fail)
            - REGULATORY GAPS: (Cite specific sections)
            - ECONOMIC IMPACT SCORE: (1-10, where 1 is pure 'Trickle-Down' and 10 is 'Shared Abundance')
            - TAX LIABILITY SUMMARY: Breakdown of {impact}
            """
            
            report = get_llm(is_cloud, st.secrets).invoke(prompt).content
            st.session_state.final_report = report
            st.error("### 📜 OFFICIAL AUDIT & ECONOMIC FINDINGS")
            st.markdown(report)
            st.download_button("📄 Download Report", create_pdf(report), file_name="ReadyAudit_Report.pdf")
            os.remove(tmp_path)

if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Ask about the 'Productivity-Distribution Equation'..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            docs = st.session_state.vector_db.similarity_search(user_input, k=15)
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {''.join([d.page_content for d in docs])}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
