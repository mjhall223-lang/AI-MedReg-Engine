def perform_gap_analysis(file_content, org_name, llm):
    """THE MECHANIC: Sifts technical files for statutory 'Holes'."""
    gap_prompt = f"""
    Perform a DEEP GAP ANALYSIS for {org_name} (Lead).
    Context: Today is March 16, 2026. The SB 25B-004 cliff is June 30, 2026.
    
    HIERARCHY CHECK:
    - Regulations/Colorado: SB 24-205 (Consumer Protection)
    - Regulations/Federal: FDA PCCP / NIST AI RMF (Affirmative Defense)
    
    Technical File Content: {file_content[:4000]}
    
    TASKS:
    1. Identify 'The Hole': Which specific statutory artifact is missing from this file?
    2. Affirmative Defense Score: How well does this architecture align with NIST AI RMF?
    3. Remediation Roadmap: Provide 3 steps to fill the hole before June 30.
    """
    return llm.invoke(gap_prompt).content
