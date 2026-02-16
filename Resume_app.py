import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's  skill search", layout="centered")

# --- 2. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I don't just search; I strategize. How can I prove Freddy is the right fit for your role?"}
    ]

# --- 3. Connections ---
@st.cache_resource
def init_connections():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        # Using Gemini 3.0 Flash Preview for high-speed reasoning
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2 # Lower temperature for better logic in agents
        )
        v_store = Milvus(
            embedding_function=embeddings,
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            },
            collection_name="RESUME_SEARCH"
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm = init_connections()

# --- 4. Deep Extraction Helper ---
def get_text(response):
    if hasattr(response, 'content'):
        return str(response.content)
    return str(response)

# --- 5. Display History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. The Agentic Logic ---
if prompt := st.chat_input("Ask about Freddy's leadership, AI, or strategy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- PHASE 1: RESEARCH PLANNING ---
        planning_prompt = f"""
        You are a Research Agent. Analyze the user question: "{prompt}"
        Break it down into 3 specific search areas to fully evaluate Freddy Goh's background.
        Example: If asked about CTO potential, search for: 1. Strategic Leadership, 2. Technical Architecture, 3. Team Mentorship.
        Output ONLY the 3 search topics, one per line.
        """
        
        with st.spinner("🧠 Agent is planning research strategy..."):
            plan_res = llm.invoke(planning_prompt)
            search_topics = [t.strip() for t in get_text(plan_res).split("\n") if t.strip()][:3]

        # --- PHASE 2: EXECUTION (TOOL USE) ---
        accumulated_context = []
        retriever = v_store.as_retriever(search_kwargs={"k": 5})
        
        for topic in search_topics:
            with st.spinner(f"🔍 Searching for: {topic}..."):
                docs = retriever.invoke(topic)
                accumulated_context.extend([d.page_content for d in docs])

        # --- PHASE 3: CRITICAL REFLECTION & ADVOCACY ---
        context_str = "\n\n".join(list(set(accumulated_context)))
        
        final_agent_prompt = f"""
        ROLE: You are Freddy Goh's Senior Career Advocate.
        
        RESEARCH DATA:
        {context_str}
        
        USER QUESTION:
        {prompt}
        
        AGENTIC TASK:
        1. Critically evaluate the research data.
        2. Identify specific achievements (metrics, years, technologies).
        3. If information is missing, use logic to explain why his senior experience (23+ years) implies capability.
        4. Draft a persuasive, executive-level recommendation.
        
        RESPONSE:
        """

        with st.spinner("⚖️ Agent is synthesizing final recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = get_text(final_res)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
