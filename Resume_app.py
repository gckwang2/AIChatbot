import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus

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
@st.cache_resource
def init_connections():
    try:
        # Check for existence of secrets first to avoid cryptic errors
        if "GOOGLE_API_KEY" not in st.secrets:
            st.error("Missing GOOGLE_API_KEY in secrets.")
            st.stop()
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", # Updated to standard model path
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Updated to the specific model you requested
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2 
        )
        
        # Verify Zilliz secrets
        if "ZILLIZ_URI" not in st.secrets or "ZILLIZ_TOKEN" not in st.secrets:
            st.error("Zilliz URI or Token missing. If you deleted your Oracle-based instance, please update these secrets.")
            st.stop()

        v_store = Milvus(
            embedding_function=embeddings,
            connection_args={
                "uri": st.secrets["ZILLIZ_URI"],
                "token": st.secrets["ZILLIZ_TOKEN"],
                "secure": True
            },
            collection_name="RESUME_SEARCH",
            drop_old=False # Ensure we don't accidentally wipe data on connect
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.info("💡 Note: If you recently deleted your Oracle Cloud resources, ensure your ZILLIZ_URI is still valid.")
        st.stop()

v_store, llm = init_connections()

# --- 4. THE CLEANER ---
def extract_clean_text(response):
    """Specifically designed to handle Gemini's complex dictionary outputs."""
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response

    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    
    return str(content)

# --- 5. Display History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. The Agentic Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- PHASE 1: Agent Research Plan ---
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        # --- PHASE 2: Execution ---
        accumulated_context = []
        # Error handling for the retriever in case the collection is empty/missing
        try:
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
            
            for topic in search_topics:
                with st.spinner(f"🔍 Searching for: {topic}..."):
                    docs = retriever.invoke(topic)
                    accumulated_context.extend([d.page_content for d in docs])
        except Exception as e:
            st.warning(f"Search failed: {e}. I will proceed with my general knowledge of Freddy.")

        # --- PHASE 3: Synthesis & Advocacy ---
        context_str = "\n\n".join(list(set(accumulated_context))) if accumulated_context else "No specific resume context found."
        
        final_agent_prompt = f"""
        ROLE: You are Freddy Goh's "Career Advocate." 
        INSTRUCTION: Analyze the resume context. Look for transferable skills and logical overlaps. 
               
        CONTEXT: {context_str}
        USER QUESTION: {prompt}
        
        TASK:
        Use the context to provide a professional, persuasive response. 
        Focus on metrics, seniority (23+ years), and leadership. 
        If a skill isn't explicitly listed, infer the related skills found in the resume, and use his previous experience based on his senior level. 
        Do not return "Career Advocate" in the return response.
        Do not show any metadata, JSON, or technical signatures.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
