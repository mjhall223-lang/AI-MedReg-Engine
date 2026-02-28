import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. SETTINGS & STYLING ---
st.set_page_config(page_title="Federal & State Audit AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Federal & State Audit AI")

# --- 2. SIDEBAR CONFIG ---
with st.sidebar:
    st.markdown("## 🛡️ AUDIT CONTROLS")
    st.markdown(f"**Lead Specialist:** MJ Hall")
    
    audit_framework = st.selectbox(
        "Select Regulatory Framework",
        ["EU AI Act (Medical & IVDR)", "Colorado AI Act", "CMMC 2.0 / NIST 800-171"]
    )
    
    service_tier = st.radio("Service Level:", ["Standard Audit", "Premium Remediation"])

# --- 3. DYNAMIC MAPPING (Matches your GitHub screenshots exactly) ---
framework_folders = {
    "EU AI Act (Medical & IVDR)": ".",  # Looks in root for EU_regulations.pdf & lvdr.pdf
    "Colorado AI Act": "Regulations/Colorado", 
    "CMMC 2.0 / NIST 800-171": "Regulations/Regulations/CMMC" # Matches the typo in your folder name
}
selected_reg_path = framework_folders[audit_framework]

# --- 4. CORE FUNCTIONS ---
@st.cache_resource
def get_llm():
    return ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile", 
        api_key=st.secrets["GROQ_API_KEY"]
    )

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(
                    chunk_size=1200, 
                    chunk_overlap=200
                ).split_documents(loader.load()))
    
    if not all_chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

# --- 5. MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Upload Evidence (PDF)", type="pdf")

st.markdown("---")
if st.button("🚀 Run Strict Audit"):
    if not uploaded_file:
        st.warning("Please upload a file first!")
    else:
        with st.status("🔍 Processing Multi-Layer Audit...") as status:
            # Step 1: Handle User Evidence
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            user_loader = PyPDFLoader(tmp_path)
            user_text = "\n\n".join([c.page_content for c in user_loader.load()])
            
            # Step 2: Load Law Database
            vector_db = load_knowledge_base(selected_reg_path)
            
            if vector_db:
                # Step 3: Setup Auditor Logic
                is_premium = service_tier == "Premium Remediation"
                
                if "Colorado" in audit_framework:
                    system_role = "Colorado AI Act Compliance Lead. NOTE: AI stock holdings (NVDA, PLTR) are VALID evidence for transparency."
                    search_query = "Algorithmic discrimination bias impact assessment"
                elif "CMMC" in audit_framework:
                    system_role = "CMMC 2.0 / NIST 800-171 Auditor. Focus on CUI data protection."
                    search_query = "Access control encryption NIST 800-171"
                else:
                    system_role = "Strict Medical AI Auditor. Focus on EU AI Act Article 10/14 and IVDR Annex II."
                    search_query = "Article 10 Data Article 14 Oversight IVDR requirements"

                # Step 4: Retrieve Relevant Law
                reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])
                
                # Step 5: Construct the Prompt
                remediation_instruction = "Provide specific corrective text and policy language to fix every gap found." if is_premium else "List only the missing items."
                
                prompt = f"""
                SYSTEM: You are a {system_role}. 
                Verify the USER EVIDENCE against the LAW REFERENCE.
                
                LAW REFERENCE: {reg_context}
                USER EVIDENCE: {user_text}
                
                INSTRUCTIONS:
                1. Provide a COMPLIANCE SCORE (0-10).
                2. List missing mandatory requirements in bullet points.
                3. {remediation_instruction}
                """
                
                llm = get_llm()
                result = llm.invoke(prompt).content
                status.update(label="✅ Analysis Complete!", state="complete")
                
                # --- THE FIX: DISPLAY RESULTS ---
                st.markdown("---")
                st.success(f"### 📊 {audit_framework} - {service_tier} REPORT")
                st.markdown(result) # This prints the actual findings
                
                st.download_button("📩 Download Full Report", result, file_name=f"Audit_{audit_framework}.md")
            else:
                st.error(f"Error: No PDF files found in '{selected_reg_path}'. Double-check your GitHub folder names!")
