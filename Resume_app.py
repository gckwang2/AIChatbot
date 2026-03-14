import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from pymilvus import connections

# 1. Critical for Streamlit + gRPC (Milvus)
nest_asyncio.apply()

# --- 2. Page Config ---
st.set_page_config(page_title="Freddy's skills finder powered by AI", layout="centered")
st.title("🚀 Freddy's Skill Search powered by AI")
st.caption("Agentic RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 3. Persistent Resources (LLM Only) ---

@st.cache_resource(show_spinner=False)
def get_llm():
    """Gemini initialization usually doesn't deadlock, so we cache it."""
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2 
    )

def get_vector_store_safe():
    """
    On-Demand Initialization. 
    This runs every time a query is made, but Milvus handle is lightweight 
    once the connection is established.
    """
    st.write("🔍 Diagnostic: Entering get_vector_store_safe...")
    
    # Check 1: Core Connection
    if not connections.has_connection("default"):
        st.write("📡 Diagnostic: Establishing Zilliz Connection...")
        connections.connect(
            alias="default",
            uri=st.secrets["ZILLIZ_URI"],
            token=st.secrets["ZILLIZ_TOKEN"],
            secure=True,
            timeout=60
        )
        st.write("✅ Diagnostic: Zilliz Connected.")
    else:
        st.write("🟢 Diagnostic: Reusing existing Zilliz connection.")
    
    # Check 2: Embeddings
    st.write("🔢 Diagnostic: Initializing Google Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    # Check 3: Object Creation (The previous hang point)
    st.write("📦 Diagnostic: Wrapping Milvus Object...")
    vstore = Milvus(
        embedding_function=embeddings,
        connection_args={"alias": "default"},
        collection_name="RESUME_SEARCH",
        search_params={"metric_type": "L2", "params": {"nprobe": 10}}
    )
    st.write("✅ Diagnostic: Vector Store Ready.")
    return vstore

# --- 4. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

# Only start the LLM at boot
with st.status("🚀 Awakening Freddy's Career Advocate...", expanded=True) as status:
    try:
        st.write("📡 Connecting to Google AI...")
        llm = get_llm()
        status.update(label="✅ Systems Online", state="complete", expanded=False)
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# --- 5. THE CLEANER ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    return str(content)

# --- 6. Display History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. The Agentic Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # PHASE 1: Agent Research Plan
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        # PHASE 2: Execution (Tool Use)
        accumulated_context = []
        
        # --- NEW SAFE LOADING BLOCK ---
        try:
            v_store = get_vector_store_safe()
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
        except Exception as e:
            st.error(f"Failed to mount database: {e}")
            st.stop()
        # ------------------------------

        for topic in search_topics:
            with st.spinner(f"🔍 Searching for: {topic}..."):
                docs = retriever.invoke(topic)
                accumulated_context.extend([d.page_content for d in docs])

        # PHASE 3: Synthesis & Advocacy
        context_str = "\n\n".join(list(set(accumulated_context)))
        
        final_agent_prompt = f"""
                ROLE: You are Freddy Goh's "Career Advocate." 
                CONTEXT: {context_str}
                USER QUESTION: {prompt}
                TASK: Use context to provide a professional, persuasive response. Do not return "Career Advocate" or technical signatures.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
