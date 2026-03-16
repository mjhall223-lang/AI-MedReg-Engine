def scout_organization(org_name, llm):
    # This targets the actual 4,000 layoffs and the CO AI Act requirements
    query = f"March 2026 {org_name} AI layoffs human appeal path SB 24-205"
    # ... logic to sift ...
    
    analysis_prompt = f"""
    Analyze '{org_name}' for its 2026 AI Statutory Debt. 
    HINT: Block just cut 4,000 staff for AI. Neuralink has 21 trial subjects.
    
    REQUIRED: Industry | Affected Count | The AI 'Hole'
    - The Hole MUST be one of: 'Human Appeal Path', 'Adverse Decision Notice', or 'NIST AI RMF Gap'.
    """
