import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf # <--- THE CONNECTION

# --- 1. CONFIG ---
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Engine")

is_cloud = "GROQ_API_KEY" in st.secrets

if is_cloud:
    st.info("🌐 **Cloud Mode** | Groq Llama-3.3")
else:
    st.success("🔒 **Local Mode** | Ollama Gemma-2")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    selected_frameworks = st.multiselect(
        "Select Framework Overlaps", 
        ["Federal Proposal (RFP Compliance)", "EU AI Act (Medical & IVDR)", "Colorado AI Act", "CMMC 2.0 (Security)", "FDA PCCP (Clinical Change)"],
        default=["Federal Proposal (RFP Compliance)"]
    )
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if "vector_db" in st.session_state: del st.session_state.vector_db
        st.rerun()

# --- 3. MAPPING ---
framework_folders = {
    "Federal Proposal (RFP Compliance)": "Regulations/Federal",
    "EU AI Act (Medical & IVDR)": "Regulations/EU",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 (Security)": "Regulations/CMMC",
    "FDA PCCP (Clinical Change)": "Regulations/FDA"
}

# --- 4. AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

if st.button("🚀 Run Multi-Framework Audit"):
    if not uploaded_file or not selected_frameworks:
        st.warning("Please upload a file and select a framework!")
    else:
        with st.status("🔍 ANALYZING REGULATORY OVERLAPS...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load context
                vector_db = load_multi_knowledge_base(selected_frameworks, framework_folders)
                
                if vector_db:
                    st.session_state.vector_db = vector_db
                    
                    # 🔥 BIGGER K: Increased to 20 for the initial audit
                    docs = vector_db.similarity_search("Definitions and mandatory requirements", k=20)
                    reg_context = "\n\n".join([f"FRAMEWORK: {d.metadata['framework']} | {d.page_content}" for d in docs])
                    
                    user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                    
                    prompt = f"""
                    SYSTEM: Global Regulatory Architect. Answer using context.
                    CONTEXT: {reg_context}
                    EVIDENCE: {user_text}
                    TASK: Status (Pass/Fail), Score (0-10), Overlap Conflicts, GAPS (Cite Sections), {'REMEDIATION: Draft language' if service_tier == 'Premium Remediation' else 'List missing items'}.
                    """
                    
                    report = get_llm(is_cloud, st.secrets).invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Audit Complete!", state="complete")
                    
                    st.error("### 📜 AUDIT FINDINGS")
                    st.markdown(report)
                    st.download_button("📄 Export PDF", create_pdf(report), file_name="Audit_Report.pdf")
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- 5. CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask about Overlaps")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_input := st.chat_input("Ex: What is the definition of High-Risk?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            # 🔥 BIGGER K: Increased to 12 for the follow-up chat
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=12)
            context_text = "\n\n".join([f"({d.metadata['framework']}) {d.page_content}" for d in context_docs])
            
            response = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
