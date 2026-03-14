import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from pymilvus import connections, utility

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's Skills Finder", layout="centered")
st.title("🚀 Freddy's Skill Search")
st.caption("Agentic RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. THE GLOBAL CONNECTION (From test-milvus.py) ---
# This ensures the 'default' connection exists before LangChain ever looks for it.
def force_milvus_connection():
    try:
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True,
                timeout=30
            )
        return True
    except Exception as e:
        st.error(f"❌ Milvus Global Connection Failed: {e}")
        return False

# Run the connection check immediately
if not force_milvus_connection():
    st.stop()

# --- 3. Initialization & Caching ---
@st.cache_resource
def init_models():
    try:
        # Verify collection exists (Sanity check from test script)
        if not utility.has_collection("RESUME_SEARCH"):
            st.error("❌ Collection 'RESUME_SEARCH' not found in Zilliz.")
            st.stop()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2
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
        st.error(f"❌ Model Initialization Failed: {e}")
        st.stop()

v_store, llm = init_models()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

# --- 4. THE CLEANER ---
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

# --- 5. UI: Sidebar Status ---
with st.sidebar:
    st.header("System Status")
    if connections.has_connection("default"):
        st.success("Milvus: Connected")
    else:
        st.error("Milvus: Disconnected")
    st.info(f"Model: Gemini 3.0 Flash")

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
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        accumulated_context = []
        try:
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
            for topic in search_topics:
                with st.spinner(f"🔍 Searching for: {topic}..."):
                    docs = retriever.invoke(topic)
                    accumulated_context.extend([d.page_content for d in docs])
        except Exception as e:
            st.warning(f"Search failed: {e}")

        context_str = "\n\n".join(list(set(accumulated_context))) if accumulated_context else "No specific resume context found."
        
        final_agent_prompt = f"""
        ROLE: You are Freddy Goh's "Career Advocate." 
        INSTRUCTION: Analyze the context provided.
        CONTEXT: {context_str}
        USER QUESTION: {prompt}
        TASK: Provide a professional, persuasive response. Do not return JSON or metadata.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
