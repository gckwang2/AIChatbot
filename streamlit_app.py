import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

# 1. Page Configuration
st.set_page_config(page_title="My Personal AI", page_icon="🛡️")
st.title("🛡️ Freddy Goh is here to help with Guardrails")

# 2. Secure API Keys (Setup these in Streamlit Cloud Secrets!)
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

# 3. Initialize the LLM (Gemini 2.0 Flash is fast and free-tier friendly)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", 
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    max_retries=3,  # Automatically retry if it hits a limit
    timeout=60      # Give it more time to respond
)

# 4. Define the Guardrail & System Persona
# This is your "Soft Guardrail" - instructions that keep the bot on track.
SYSTEM_PROMPT = """
You are a helpful personal assistant. 
GUARDRAILS:
- If the user asks about illegal activities, say 'I cannot assist with that.'
- If the user asks about topics outside of personal productivity or your data, politely redirect them.
- Always be concise and professional.
"""

# 5. Initialize the LangGraph Agent
# 'tools=[]' is where you will add your Vector DB search tool later.

agent_executor = create_react_agent(
    model, 
    tools=[], 
    prompt=SYSTEM_PROMPT 
)

# 6. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Chat Input & Execution
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Run the LangGraph agent
                response = agent_executor.invoke(
                    {"messages": [HumanMessage(content=prompt)]}
                )
                
                # Extract the final message from the graph
                final_answer = response["messages"][-1].content
                
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")
