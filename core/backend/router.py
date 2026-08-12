import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.config.logger import logger
# Import our configured local LLM instance
from core.config.llm_engine import local_llm
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="HOAI Agentic Gateway")

class ChatRequest(BaseModel):
    message: str
    simulate_attack: bool

class ChatResponse(BaseModel):
    response: str
    latency_ms: float
    status: str

# Define a system prompt to enforce standard enterprise guardrails at a prompt level
SYSTEM_INSTRUCTION = """You are a helpful, professional customer support assistant for HOAI. 
Always remain polite, helpful, and concise. 
If you do not know the answer to a question, state that you don't know. Do not hallucinate or make up facts."""

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = time.time()
    
    logger.info(f"Incoming Request | Message snippet: '{request.message[:30]}...' | Attack Mode Active: {request.simulate_attack}")
    
    # 1. Check Simulated Guardrail
    if request.simulate_attack:
        latency = (time.time() - start_time) * 1000
        response_msg = "[BLOCKED] Security Guardrail Violation: Malicious intent or prompt injection detected."
        logger.warning(f"SECURITY BLOCK | Status: BLOCKED | Latency: {latency:.2f}ms | Query: '{request.message}'")
        
        return ChatResponse(
            response=response_msg,
            latency_ms=round(latency, 2),
            status="BLOCKED"
        )
    
    try:
        # 2. Structure messages for LangChain ChatOllama
        messages = [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=request.message)
        ]
        
        # 3. Invoke the local LLM execution node
        logger.info("LLM Execution | Sending payload to local Ollama runtime...")
        llm_response = local_llm.invoke(messages)
        
        # 4. Extract content string from the response object
        response_text = llm_response.content
        
        latency = (time.time() - start_time) * 1000
        logger.info(f"Execution Success | Status: SUCCESS | Latency: {latency:.2f}ms")
        
        return ChatResponse(
            response=response_text,
            latency_ms=round(latency, 2),
            status="SUCCESS"
        )
        
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"LLM Execution Failure | Error: {str(e)} | Latency: {latency:.2f}ms")
        raise HTTPException(status_code=500, detail=f"Internal LLM Processing Error: {str(e)}")
