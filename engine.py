import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_knowledge_base(path):
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load()))
    
    if not all_chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

def perform_audit(user_text, vector_db, framework, llm):
    # Retrieve law context based on the framework
    search_query = "Impact assessment bias" if "Colorado" in framework else "Article 10"
    reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])

    # Setup the Auditor Personality
    if "Colorado" in framework:
        system_role = "Colorado AI Act Auditor. AI stock holdings (NVDA, PLTR) are VALID evidence for transparency."
    elif "CMMC" in framework:
        system_role = "CMMC 2.0 Auditor. Focus on NIST 800-171 and CUI security."
    else:
        system_role = "Medical AI Auditor. Focus on Article 10/14."

    prompt = f"""
    SYSTEM: You are a {system_role}. 
    Audit the USER EVIDENCE against the LAW provided.
    
    LAW: {reg_context}
    USER EVIDENCE: {user_text}
    
    OUTPUT: Provide a PASS/FAIL score and list MISSING items.
    """
    return llm.invoke(prompt).content
