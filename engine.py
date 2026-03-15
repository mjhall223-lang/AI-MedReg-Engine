import os
import requests
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
# ... existing imports ...

def find_and_scrape_company(company_name):
    """Uses web search to find the company's AI disclosure or Terms of Service."""
    search = TavilySearchResults(k=2)
    # Search for specific regulatory keywords
    query = f"{company_name} AI transparency policy terms of service IVDR compliance"
    results = search.run(query)
    
    scraped_content = ""
    for res in results:
        try:
            url = res['url']
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from paragraphs to simulate the 'Evidence' PDF
            scraped_content += "\n".join([p.get_text() for p in soup.find_all('p')])
        except:
            continue
    return scraped_content[:15000] # Limit context for the LLM
