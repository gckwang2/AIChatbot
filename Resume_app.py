import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
# --- FIX 1: Import connections from pymilvus ---
from pymilvus import connections 

# --- FIX 2: Apply nest_asyncio to prevent gRPC deadlocks ---
nest_asyncio.apply()

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's skills finder powered by AI", layout="centered")
st.title("🚀 Freddy's Skill Search powered by AI")
st.caption("Agentic RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

# --- 3. Connections ---
@st.cache_resource(show_spinner=False)
def get_vector_store():
    # This now works because 'connections' is imported
    if not connections.has_connection("default"):
        connections.connect(
            alias="default",
            uri=st.secrets["ZILLIZ_URI"],
            token=st.secrets["ZILLIZ_TOKEN"],
            secure=True,
            timeout=30
        )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    return Milvus(
        embedding_function=embeddings,
        connection_args={"alias": "default"},
        collection_name="RESUME_SEARCH",
        search_params={"metric_type": "L2", "params": {"nprobe": 10}}
    )

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2 
    )

# --- 4. Execution ---
# We use the status box to call the cached functions
with st.status("🚀 Awakening Freddy's Career Advocate...", expanded=False) as status:
    v_store = get_vector_store()
    llm = get_llm()
    status.update(label="✅ Systems Online", state="complete")

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
        # PHASE 1: Planning
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        # PHASE 2: Execution
        accumulated_context = []
        retriever = v_store.as_retriever(search_kwargs={"k": 5})
        
        for topic in search_topics:
            with st.spinner(f"🔍 Searching: {topic}..."):
                docs = retriever.invoke(topic)
                accumulated_context.extend([d.page_content for d in docs])

        # PHASE 3: Synthesis
        context_str = "\n\n".join(list(set(accumulated_context)))
        
        final_agent_prompt = f"""
                ROLE: You are Freddy Goh's "Career Advocate." 
                INSTRUCTION: Analyze the resume context provided. 
               
        CONTEXT: {context_str}
        USER QUESTION: {prompt}
        
        TASK:
        Use the context to provide a professional, persuasive response. 
        Focus on metrics, seniority (23+ years), and leadership. 
        If a skill isn't explicitly listed, infer the related skills.
        Do not show any metadata, JSON, or technical signatures.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
