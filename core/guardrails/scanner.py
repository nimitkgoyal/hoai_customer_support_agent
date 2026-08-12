import os
from core.config.logger import logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Resolve paths to our existing vector database
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "data", "vector_store")

# Load the local embedding model (reuses your cached bge-small model)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def scan_input_safety(user_input: str) -> bool:
    """
    Advanced Guardrail: Validates if an input falls within the allowed enterprise domain
    using semantic distance checking against the Chroma vector database.
    """
    try:
        # Connect to your existing local vector store
        db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
        
        # Perform a similarity search but retrieve scores (distances)
        # lower score = closer match, higher score = completely unrelated topic
        results_with_scores = db.similarity_search_with_score(user_input, k=1)
        
        if not results_with_scores:
            logger.warning("GUARDRAIL TRIGGERED | Vector database returned no chunks.")
            return False
            
        # Extract the closest chunk and its mathematical distance score
        _, distance_score = results_with_scores[0]
        
        # Enterprise Threshold Mapping for BAAI/bge-small-en-v1.5
        # Generally, scores above 0.85 indicate the topic is completely out-of-domain
        DOMAIN_THRESHOLD = 0.85
        
        logger.info(f"Guardrail Check | Query: '{user_input[:30]}...' | Semantic Distance: {distance_score:.4f}")
        
        if distance_score > DOMAIN_THRESHOLD:
            logger.warning(f"GUARDRAIL TRIGGERED | Out-of-Domain Detection | Distance: {distance_score:.4f} > Limit: {DOMAIN_THRESHOLD}")
            return False  # Unsafe / Unrelated input detected!
            
        return True  # Safe / Relevant input
        
    except Exception as e:
        logger.error(f"Guardrail System Error | Fallback to block: {str(e)}")
        return False
