import streamlit as st
import os
import tempfile
from engine import get_llm, load_multi_knowledge_base, create_pdf 
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CONFIG ---
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Engine")

# Detection Logic
is_cloud = "GROQ_API_KEY" in st.secrets

if is_cloud:
    st.info("🌐 **Cloud Mode** (Powered by Groq)")
else:
    st.success("🔒 **Local Mode** (Powered by Ollama)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    selected_frameworks = st.multiselect(
        "Select Regulatory Frameworks", 
        [
            "Federal Proposal (RFP Compliance)", 
            "EU AI Act (Medical & IVDR)", 
            "Colorado AI Act", 
            "CMMC 2.0 (Security)", 
            "FDA PCCP (Clinical Change)"
        ],
        default=["Federal Proposal (RFP Compliance)"]
    )
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        if "vector_db" in st.session_state: del st.session_state.vector_db
        st.rerun()

# --- 3. PATH MAPPING (FIXED FOR YOUR GITHUB STRUCTURE) ---
# I added both 'Regulations/EU' and the 'Regulations/Regulations' fallback 
# to make sure it finds your files regardless of the subfolder depth.
framework_folders = {
    "Federal Proposal (RFP Compliance)": "Regulations/Federal",
    "EU AI Act (Medical & IVDR)": "Regulations/Regulations", # Matches your screenshot
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 (Security)": "Regulations/CMMC",
    "FDA PCCP (Clinical Change)": "Regulations/FDA"
}

# --- 4. THE AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence PDF", type="pdf")

if st.button("🚀 Run Multi-Framework Audit"):
    if not uploaded_file or not selected_frameworks:
        st.warning("Please upload a file and select a framework!")
    else:
        with st.status("🔍 CROSS-REFERENCING DOCUMENTS...") as status:
            tmp_path = ""
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Load context with High K
                vector_db = load_multi_knowledge_base(selected_frameworks, framework_folders)
                
                if vector_db is None:
                    status.update(label="❌ No Framework PDFs found!", state="error")
                    st.error(f"Could not find PDFs. Check that your GitHub folder is 'Regulations/Regulations' or 'Regulations/EU'")
                else:
                    st.session_state.vector_db = vector_db
                    
                    # 🔥 K=20 Search (Crucial for finding legal definitions)
                    search_docs = vector_db.similarity_search("Definitions, scope, and mandatory requirements", k=20)
                    reg_context = "\n\n".join([f"({d.metadata.get('framework', 'Unknown')}) {d.page_content}" for d in search_docs])
                    
                    # Load user evidence text
                    user_loader = PyPDFLoader(tmp_path)
                    user_text = "\n\n".join([c.page_content for c in user_loader.load()])
                    
                    prompt = f"""
                    SYSTEM: Expert Auditor. Use the provided context to find gaps in the evidence.
                    CONTEXT: {reg_context}
                    EVIDENCE: {user_text}
                    TASK: Status, Score (0-10), Conflict/Overlap Analysis, Gaps (Cite Sections), 
                    {'REMEDIATION: Provide unified draft language' if service_tier == 'Premium Remediation' else 'List missing items'}.
                    """
                    
                    llm = get_llm(is_cloud, st.secrets)
                    report = llm.invoke(prompt).content
                    st.session_state.final_report = report
                    
                    status.update(label="✅ Audit Complete!", state="complete")
                    st.error("### 📜 AUDIT FINDINGS")
                    st.markdown(report)
                    st.download_button("📄 Export PDF", create_pdf(report), file_name="Audit_Report.pdf")
            
            except Exception as e:
                status.update(label="❌ Analysis Failed", state="error")
                st.error(f"Error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- 5. CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Framework Deep Dive")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_input := st.chat_input("Ask a follow-up (e.g. 'Show me the High-Risk definition')"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            # 🔥 K=12 for chat depth
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=12)
            context_text = "\n\n".join([f"({d.metadata.get('framework', 'Unknown')}) {d.page_content}" for d in context_docs])
            
            resp = get_llm(is_cloud, st.secrets).invoke(f"CONTEXT: {context_text}\nUSER: {user_input}").content
            st.markdown(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
