import os
from pathlib import Path
from langchain_groq import ChatGroq

def get_llm(st_secrets):
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def list_all_laws(base_dir="Regulations"):
    """Recursively finds all PDF laws in your nested folder structure."""
    laws = []
    # .rglob('*') handles your 'Regulations/Regulations' nesting automatically
    path_root = Path(base_dir)
    for file in path_root.rglob('*.pdf'):
        # We store the relative path so the tool knows exactly which folder it's in
        laws.append(str(file.relative_to(path_root.parent)))
    return sorted(laws)

def perform_gap_analysis(file_content, selected_laws, llm):
    """Audits the uploaded file against ONLY the toggled laws."""
    # Logic to read the selected_laws text and compare
    laws_context = f"Auditing against: {', '.join(selected_laws)}"
    
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS.
    Context: {laws_context}
    Target Architecture: {file_content[:3000]}
    
    Identify specific missing compliance artifacts required by these laws.
    """
    return llm.invoke(gap_prompt).content
