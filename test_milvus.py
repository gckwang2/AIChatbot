import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from pymilvus import connections, utility

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's skills finder", layout="centered")
st.title("🚀 Freddy's Skill Search")

# --- 2. GLOBAL CONNECTION (The Fix) ---
# This MUST run on every script rerun to ensure the 'default' alias exists
try:
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            uri=st.secrets["ZILLIZ_URI"],
            token=st.secrets["ZILLIZ_TOKEN"],
            secure=True
        )
except Exception as e:
    st.error(f"❌ Global Connection Failed: {e}")
    st.stop()

# --- 3. Cached Resource Loading ---
@st.cache_resource
def load_models():
    try:
        # Verification check
        if not utility.has_collection("RESUME_SEARCH"):
            st.error("❌ Collection 'RESUME_SEARCH' not found.")
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
        st.error(f"❌ Model Loading Failed: {e}")
        st.stop()

v_store, llm = load_models()

# --- 4. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

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
        CONTEXT: {context_str}
        USER QUESTION: {prompt}
        TASK: Use the context to provide a professional, persuasive response. Do not return "Career Advocate" or JSON.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
