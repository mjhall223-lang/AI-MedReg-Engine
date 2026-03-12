import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf 
from langchain_community.document_loaders import PyPDFLoader

# --- CONFIG ---
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    # 1. Added "Federal Proposal" as a default to include your new PDFs
    selected_frameworks = st.multiselect(
        "Active Frameworks", 
        ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0", "FDA PCCP"],
        default=["Federal Proposal", "EU AI Act", "Colorado AI Act"]
    )
    
    # 2. Reference Library Monitor (Visual confirmation of your uploads)
    with st.expander("📚 Reference Library"):
        st.caption("Active Federal Docs:")
        st.code("WH_AI_Great_Divergence_2026.pdf\nTX_CPA_AI_Tax_Compliance_P2.pdf")

    service_tier = st.radio("Audit Depth:", ["Standard Scan", "Premium Remediation"])
    
    if st.button("🗑️ Reset Engine"):
        st.session_state.messages = []
        if "vector_db" in st.session_state: del st.session_state.vector_db
        if "final_report" in st.session_state: del st.session_state.final_report
        st.rerun()

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload Evidence PDF for Auditing", type="pdf")

if st.button("🚀 Run Multi-Framework Audit"):
    if not uploaded_file:
        st.warning("Please upload a document to audit!")
    else:
        with st.status("🔍 GATHERING REGULATORY CONTEXT...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load context using our Smart-Scan
                vector_db = load_multi_knowledge_base(selected_frameworks)
                
                if vector_db is None:
                    status.update(label="❌ No Regulation PDFs Found!", state="error")
                else:
                    st.session_state.vector_db = vector_db
                    
                    # 3. Expanded search query to include the new Federal Tax/Economic logic
                    search_query = """
                    Definitions of High-Risk, Section 1701, Article 6, 
                    AI tax compliance, digital payroll taxes, and labor displacement metrics
                    """
                    search_docs = vector_db.similarity_search(search_query, k=25)
                    reg_context = "\n\n".join([f"(File: {d.metadata.get('source_file')}) {d.page_content}" for d in search_docs])
                    
                    user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                    
                    # 4. Enhanced System Instruction with Federal Tax Logic
                    prompt = f"""
                    SYSTEM: You are a Senior Regulatory Architect and Economic Policy Analyst. 
                    CONTEXT: {reg_context}
                    EVIDENCE: {user_text}

                    INSTRUCTIONS:
                    - Search specifically for definitions in Colorado SB 24-205 Section 6-1-1701.
                    - Search specifically for EU AI Act Article 6.
                    - Identify potential "Digital Payroll Tax" liabilities or "Great Divergence" labor impacts based on the Federal Proposal docs.
                    - Compare the evidence against these specific definitions and economic frameworks.
                    
                    OUTPUT:
                    1. STATUS: (Pass/Fail)
                    2. SCORE: (0-10)
                    3. GAPS: (Cite legal sections AND potential tax/economic liabilities)
                    4. REMEDIATION: {'Provide specific draft language' if service_tier == 'Premium Remediation' else 'List missing items'}.
                    """
                    
                    report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Audit Complete!", state="complete")
                    st.error("### 📜 OFFICIAL AUDIT FINDINGS")
                    st.markdown(report)
                    st.download_button("📄 Download Audit Report", create_pdf(report), file_name="ReadyAudit_Report.pdf")
            
            except Exception as e:
                status.update(label="❌ Error Encountered", state="error")
                st.error(str(e))
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Interactive Regulatory Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_input := st.chat_input("Ask about tax compliance or follow-ups..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=15)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
