import re
from duckduckgo_search import DDGS 

def get_llm(st_secrets):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=st_secrets["GROQ_API_KEY"])

def scout_organization(org_name, llm):
    # Targets the 4,000 layoffs and the specific CO AI Act triggers
    query = f"March 2026 {org_name} AI layoffs human appeal path SB 24-205"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            news_text = "\n\n".join([f"{r['title']}: {r['body']}" for r in results])
    except: news_text = "Search offline."

    analysis_prompt = f"""
    Analyze '{org_name}' for its 2026 AI Statutory Debt. 
    HINT: Block just cut 4,000 staff for AI. Neuralink has 21 trial subjects.
    
    REQUIRED: Industry | Affected Count | The AI 'Hole'
    - The Hole MUST be one of: 'Human Appeal Path', 'Adverse Decision Notice', or 'NIST AI RMF Gap'.
    """
    analysis = llm.invoke(analysis_prompt).content
    return news_text, analysis
