from langchain_ollama import ChatOllama
from core.config.logger import logger

try:
    # Initialize the local Ollama connection
    # We set temperature=0.0 for deterministic, predictable enterprise support outputs
    local_llm = ChatOllama(
        model="llama3.2:3b", 
        temperature=0.0,
        num_ctx=4096       # Sets a 4k token context window
    )
    logger.info("LLM Engine | Successfully initialized local ChatOllama client.")
except Exception as e:
    logger.error(f"LLM Engine | Failed to initialize ChatOllama: {str(e)}")
    raise e
