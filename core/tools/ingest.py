import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define structured paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
POLICY_FILE = os.path.join(DATA_DIR, "knowledge_base", "company_policy.md")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_store")

def run_ingestion():
    print("🚀 Starting Vector Ingestion Pipeline...")
    
    if not os.path.exists(POLICY_FILE):
        print(f"❌ Error: Source file not found at {POLICY_FILE}")
        return

    # 1. Read Markdown Data
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # 2. Chunking strategy: Split cleanly by Markdown Headers for structural relevance
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = markdown_splitter.split_text(markdown_text)
    print(f"📦 Document split successfully into {len(chunks)} contextual chunks.")

    # 3. Load Embeddings Engine (Runs locally on CPU)
    print("🧠 Initializing Local Embedding Model (bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 4. Generate and Store Vectors in Local Storage
    print(f"💾 Saving vector embeddings to local database at: {VECTOR_DB_DIR}")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print("✅ Ingestion Pipeline Complete! Local vector database is ready.")

if __name__ == "__main__":
    run_ingestion()
