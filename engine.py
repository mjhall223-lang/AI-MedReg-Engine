import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama 

def load_knowledge_base(path):
    """Loads PDFs from your 'Regulations' folder into a private database."""
    all_chunks = []
    if os.path.exists(path):
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(path, f))
                # Breaks long legal docs into smaller pieces for the AI to read
                all_chunks.extend(RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=100
                ).split_documents(loader.load()))
    
    if not all_chunks:
        return None
        
    # This turns text into math on your Chromebook (stays local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(all_chunks, embeddings)

def perform_audit(user_text, vector_db, framework, llm):
    """The main engine that finds gaps and scores your work."""
    
    # Decide what to search for based on the job
    if "Federal" in framework:
        search_query = "RFP Section L Section M compliance requirements"
    else:
        search_query = "Article 10 data governance human oversight"
        
    # Find the most relevant parts of the law/RFP
    reg_context = "\n\n".join([d.page_content for d in vector_db.similarity_search(search_query, k=5)])

    # Set the AI's "Job Title"
    if "Federal" in framework:
        system_role = "Senior Federal Proposal Manager. Focus on 100% RFP compliance."
    else:
        system_role = "Medical AI Auditor. Focus on Article 10/14."

    # The instructions for the AI
    prompt = f"""
    SYSTEM: You are a {system_role}. 
    Audit the PROPOSAL DRAFT against the RFP REQUIREMENTS provided.
    
    RFP REQUIREMENTS: {reg_context}
    PROPOSAL DRAFT: {user_text}
    
    OUTPUT: 
    1. STATUS: (Compliant, Partially Compliant, or Non-Compliant)
    2. SCORE: (Rank out of 10)
    3. GAPS: (List every missing requirement or 'shall' statement)
    4. WIN THEME: (Suggest one way to make this more compelling)
    """
    return llm.invoke(prompt).content

# --- This part tells the app to use the local Gemma brain on your Chromebook ---
# When you run your Streamlit_app.py, make sure it calls:
# llm = ChatOllama(model="gemma2:2b")
