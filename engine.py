import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    # Use the 2026-spec Llama 3.3 for high-stakes policy sifting
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def perform_gap_analysis(file_content, org_name, llm):
    """THE MECHANIC: Sifts technical files for statutory 'Holes'."""
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS for {org_name}.
    Context: Today is March 16, 2026. The SB 25B-004 cliff is June 30, 2026.
    
    HIERARCHY CHECK:
    - Regulations/Colorado: SB 24-205 (Consumer Protection)
    - Regulations/Federal: FDA PCCP / NIST AI RMF (Affirmative Defense)
    
    Technical File Content: {file_content[:4000]}
    
    TASKS:
    1. Identify 'The Hole': Which specific statutory artifact is missing?
    2. Affirmative Defense Score: Alignment with NIST AI RMF.
    3. Remediation: 3 steps to fill the hole before June 30.
    """
    return llm.invoke(gap_prompt).content
