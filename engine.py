import re

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def perform_gap_analysis(file_content, org_name, count, llm):
    """THE MECHANIC: Compares uploaded tech files against your DB hierarchy."""
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS for {org_name}.
    Target: {count} AI-driven employment decisions (Risk: ${count * 20000:,}).
    
    FOLDER CONTEXT (Regulations/Regulations/):
    - Colorado: SB 24-205 & SB 25B-004 (June 30, 2026 Cliff)
    - NIST: SP 800-171 & AI RMF (Affirmative Defense)
    
    Technical File Content: {file_content[:4000]}
    
    TASKS:
    1. Identify MISSING ARTIFACTS: Human Appeal Path, Adverse Decision Notice, and Annual Impact Assessment.
    2. Score 'Reasonable Care' alignment with NIST AI RMF.
    3. List remediation steps to be completed by June 30, 2026.
    """
    return llm.invoke(gap_prompt).content
