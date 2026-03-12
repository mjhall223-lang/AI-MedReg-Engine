import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf, EconomicImpact
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework & Economic Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
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
    st.caption("Predict 'Robot Tax' Liabilities")
    est_tokens = st.number_input("Est. Monthly Tokens (Millions):", min_value=0.0, value=10.0)
    est_replaced = st.number_input("Est. Human Roles Replaced:", min_value=0, value=1)
    
    impact = EconomicImpact.calculate_liability(token_usage=est_tokens*1000000, replaced_staff=est_replaced)
    
    st.metric("Total Tax Liability", f"${impact['total']:,}")
    with st.expander("View Breakdown"):
        st.write(f"Token Tax: ${impact['token_tax']:,}")
        st.write(f"Digital Payroll: ${impact['payroll_tax']:,}")

    if st.button("🗑️ Reset Engine"):
        st.session_state.clear()
        st.rerun()

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Comprehensive Audit"):
    if not uploaded_file:
        st.warning("Please upload a document!")
    else:
        with st.status("🔍 ANALYZING REGULATORY & ECONOMIC RISK...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                vector_db = load_multi_knowledge_base(selected_frameworks)
                st.session_state.vector_db = vector_db
                
                # Context Search
                search_docs = vector_db.similarity_search("AI tax liability, labor displacement, productivity pay gap, Section 1701", k=25)
                reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
                user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                
                prompt = f"""
                SYSTEM: You are a Senior Regulatory Architect & AI Economic Consultant.
                CONTEXT: {reg_context}
                EVIDENCE: {user_text}
                ECONOMIC ESTIMATE: {impact}

                INSTRUCTIONS:
                1. Audit for EU AI Act / Colorado AI Act compliance.
                2. Evaluate if this AI usage falls under the new 'Robot Tax' or 'AI Dividend' liability.
                3. Address the 'Great Divergence'—does this system benefit the worker or just the company?

                OUTPUT:
                - STATUS: (Pass/Fail)
                - REGULATORY GAPS: (Cite legal sections)
                - ECONOMIC IMPACT SCORE: (1-10, where 1 is pure 'Trickle-Down' greed and 10 is 'Abundance-Shared')
                - TAX LIABILITY SUMMARY: Mention the estimate from {impact}
                """
                
                report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                st.session_state.final_report = report
                status.update(label="✅ Audit Complete!", state="complete")
                st.error("### 📜 OFFICIAL AUDIT & ECONOMIC FINDINGS")
                st.markdown(report)
                st.download_button("📄 Download Audit Report", create_pdf(report), file_name="ReadyAudit_Full_Report.pdf")
            
            except Exception as e: st.error(str(e))
            finally: 
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    if user_input := st.chat_input("Ask about the 'Productivity-Distribution Equation'..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=15)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
