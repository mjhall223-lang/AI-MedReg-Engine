import os
import sys
import site
import torch
from importlib import reload

# 1. SETUP & REFRESH
print("🛠️ Polishing the engine...")
!pip uninstall -y -q google-colab bigframes
!pip install -q -U langchain-huggingface langchain-community pypdf langchain-text-splitters faiss-cpu sentence-transformers transformers accelerate bitsandbytes
reload(site)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# 2. DATA INGESTION
print("📚 Mapping the EU AI Act...")
dataset_path = "/kaggle/input/european-ai-device-regs"
all_docs = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(root, file))
            all_docs.extend(loader.load())

if not all_docs:
    print("❌ ERROR: No PDFs found. Check your dataset path.")
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print(f"🚀 ENGINE ONLINE: {len(chunks)} legal segments indexed.")

    # 3. THE AUDITOR BRAIN (Qwen-2.5-1.5B)
    print("🧠 Booting Senior Auditor Brain...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000, temperature=0.1)
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # 4. THE HIGH-PRECISION AUDIT LOGIC
    def perform_audit(query):
        docs = vector_db.similarity_search(query, k=6)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = f"""<|im_start|>system
You are a Lead AI Regulatory Auditor for Medical Devices. 
Ground every answer in Article 10 (Data Governance) and Article 14 (Human Oversight).

FORMAT:
🔴 RED: Missing a specific mandatory requirement.
🟡 YELLOW: Mentioned but lacks technical validation or metrics.
🟢 GREEN: Pass.<|im_end|>
<|im_start|>user
LEGAL CONTEXT:
{context}

AUDIT QUESTION: {query}<|im_end|>
<|im_start|>assistant
REPORT:"""
        return local_llm.invoke(prompt)

    # 5. TEST WITH DUMMY DATA
    print("\n--- [FOUNDER MODE] RUNNING DUMMY DATA TEST ---")
    dummy_client_data = """
    TECHNICAL SUMMARY: 'BioScan AI Diagnostic'
    - Intended Use: Diagnostic aid for identifying skin conditions from smartphone photos.
    - Data Collection: We downloaded 50,000 photos from a public research dataset on Kaggle. 
    - Processing: Images were resized and normalized. 
    - Oversight: The doctor can review the results on a dashboard once a week.
    """
    
    test_query = f"Audit this summary against Article 10 bias and Article 14 human oversight requirements:\n\n{dummy_client_data}"
    print(perform_audit(test_query))
