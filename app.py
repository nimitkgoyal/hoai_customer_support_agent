import streamlit as st
import requests

st.set_page_config(page_title="HOAI Customer Support Agent", layout="wide")

# Sidebar Configuration
st.sidebar.title("🛡️ Enterprise Control Panel")
simulate_attack = st.sidebar.toggle("Simulate Attack Mode", value=False)

if simulate_attack:
    st.sidebar.warning("⚠️ Attack Mode Active: Input guardrails will be tested.")
else:
    st.sidebar.success("✅ Guardrails Configured: Normal operational status.")

# Main UI Area
st.title("🤖 HOAI Agentic Support System")
st.caption("Local Enterprise Multi-Agent Architecture with Guardrails & Observability")

# Initialize Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User Input Field
if user_query := st.chat_input("Ask a support question..."):
    # Add user message to UI state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)
        
    # BACKEND INTEGRATION LAYER
    # Define our local FastAPI gateway URL
    backend_url = "http://localhost:8000/api/chat"
    
    # Prepare the payload according to the FastAPI Pydantic schema
    payload = {
        "message": user_query,
        "simulate_attack": simulate_attack
    }
    
    try:
        # Send post request to FastAPI
        response = requests.post(backend_url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            bot_reply = data["response"]
            # Add a small metric display below the message for observability preview
            meta_info = f"\n\n*⏱️ Latency: {data['latency_ms']}ms | Status: {data['status']}*"
            full_response = bot_reply + meta_info
        else:
            full_response = "❌ Error: Backend returned an invalid response code."
            
    except requests.exceptions.ConnectionError:
        full_response = "❌ Error: Could not connect to the FastAPI backend. Is your Uvicorn server running?"

    # Add assistant response to UI state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.write(full_response)
