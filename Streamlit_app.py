import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf 
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Engine")

is_cloud = "GROQ_API_KEY" in st.secrets
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    selected_frameworks = st.multiselect(
        "Select Regulatory Frameworks", 
        ["Federal Proposal", "EU AI Act", "Colorado AI Act", "CMMC 2.0", "FDA PCCP"],
        default=["EU AI Act"]
    )
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        if "vector_db" in st.session_state: del st.session_state.vector_db
        st.rerun()

uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Multi-Framework Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 SCANNING REPOSITORY FOR REGULATIONS...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # --- SMART SCAN ---
                # This now looks in 'Regulations/' and finds everything inside it
                vector_db = load_multi_knowledge_base(selected_frameworks, root_folder="Regulations")
                
                if vector_db is None:
                    status.update(label="❌ No PDFs found in /Regulations folder!", state="error")
                else:
                    st.session_state.vector_db = vector_db
                    docs = vector_db.similarity_search("Definitions and requirements", k=20)
                    reg_context = "\n\n".join([f"(Source: {d.metadata.get('source_file')}) {d.page_content}" for d in docs])
                    
                    user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                    
                    prompt = f"CONTEXT: {reg_context}\nEVIDENCE: {user_text}\nTASK: Provide Status, Score, Gaps, and {'Remediation' if service_tier == 'Premium Remediation' else 'Missing Items'}."
                    
                    report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Audit Complete!", state="complete")
                    st.markdown(report)
                    st.download_button("📄 Export PDF", create_pdf(report), file_name="Audit_Report.pdf")
            except Exception as e:
                status.update(label="❌ Error", state="error")
                st.error(str(e))
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Framework Deep Dive")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if user_input := st.chat_input("Ask a follow-up..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=12)
            context_text = "\n\n".join([d.page_content for d in context_docs])
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
