import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fpdf import FPDF

# --- 1. CONFIG & SESSION STATE ---
st.set_page_config(page_title="ReadyAudit Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ReadyAudit: Multi-Framework Engine")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown("**Lead Specialist:** Myia Hall")
    
    # CHANGED: Select multiple frameworks for overlap analysis
    selected_frameworks = st.multiselect(
        "Select Framework Overlaps", 
        [
            "EU AI Act (Medical & IVDR)", 
            "Colorado AI Act", 
            "CMMC 2.0",
            "Medical Bias & Health Equity",
            "FDA PCCP (Clinical Change)"
        ],
        default=["FDA PCCP (Clinical Change)"]
    )
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 2. GITHUB MAPPING ---
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0": "Regulations/Regulations/CMMC",
    "Medical Bias & Health Equity": ".",
    "FDA PCCP (Clinical Change)": "Regulations/Regulations/Federal"
}

# --- 3. CORE FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])

def load_multi_knowledge_base(selected_list):
    all_chunks = []
    for framework in selected_list:
        path = framework_folders[framework]
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(path, f))
                    # Tagging chunks with their source framework for better "Overlap" citations
                    docs = loader.load()
                    for d in docs:
                        d.metadata["framework"] = framework
                    all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs))
    
    if not all_chunks: return None
    return FAISS.from_documents(all_chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "OFFICIAL MULTI-FRAMEWORK AUDIT", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 10, f"Lead Specialist: Myia Hall | Frameworks: {', '.join(selected_frameworks)}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    safe_text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, txt=safe_text)
    return bytes(pdf.output())

# --- 4. AUDIT ENGINE ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

if st.button("🚀 Run Multi-Framework Audit"):
    if not uploaded_file or not selected_frameworks:
        st.warning("Please upload a file and select at least one framework!")
    else:
        with st.status("🔍 ANALYZING OVERLAPS ACROSS SELECTED REGIMES...") as status:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                user_text = "\n\n".join([c.page_content for c in PyPDFLoader(tmp_path).load()])
                vector_db = load_multi_knowledge_base(selected_frameworks)
                
                if vector_db:
                    st.session_state.vector_db = vector_db
                    
                    # Retrieve context from ALL selected frameworks
                    docs = vector_db.similarity_search("Mandatory requirements, missing document gaps, and technical pillars", k=8)
                    reg_context = "\n\n".join([f"FRAMEWORK: {d.metadata['framework']} | SOURCE: {os.path.basename(d.metadata['source'])}:\n{d.page_content}" for d in docs])
                    
                    prompt = f"""
                    SYSTEM: You are a Global Regulatory Architect. You are auditing evidence against MULTIPLE frameworks.
                    
                    FRAMEWORKS SELECTED: {', '.join(selected_frameworks)}
                    
                    INTERNAL STANDARDS & LAWS: 
                    {reg_context}
                    
                    EVIDENCE: 
                    {user_text}

                    TASK:
                    1. STATUS: Provide a status for EACH selected framework.
                    2. SCORE: Provide an aggregate score [0-10]. 
                    3. OVERLAP ANALYSIS: Identify if a gap in one framework (e.g., FDA) also creates a failure in another (e.g., EU AI Act).
                    4. GAPS: Cite specific requirements.
                    5. {"REMEDIATION: Provide unified draft language that satisfies ALL selected frameworks." if service_tier == "Premium Remediation" else "List missing requirements."}
                    """
                    
                    report = get_llm().invoke(prompt).content
                    st.session_state.final_report = report
                    status.update(label="✅ Global Audit Complete!", state="complete")
                    
                    st.error("### 📜 GLOBAL AUDIT FINDINGS")
                    st.markdown(report)
                    st.download_button("📄 Export Official Multi-Audit PDF", create_pdf(report), file_name="Global_Audit.pdf", mime="application/pdf")
            finally:
                if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

# --- 5. INTERACTIVE OVERLAP CHAT ---
if "final_report" in st.session_state:
    st.markdown("---")
    st.subheader("💬 Ask about Framework Overlaps")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])

    if user_input := st.chat_input("Ex: Does my FDA Traceability Table satisfy the EU IVDR requirements?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)

        with st.chat_message("assistant"):
            context_docs = st.session_state.vector_db.similarity_search(user_input, k=4)
            context_text = "\n\n".join([f"({d.metadata['framework']}) {d.page_content}" for d in context_docs])
            
            response = get_llm().invoke(f"CONTEXT: {context_text}\nQUESTION: {user_input}\nExplain the overlap and how to satisfy multiple frameworks at once.").content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
