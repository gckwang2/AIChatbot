import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
# IMPORT: This is the key to managing the connection list manually
from pymilvus import connections, utility

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's skills finder", layout="centered")
st.title("🚀 Freddy's Skill Search")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        # 1. Check if "default" connection already exists to prevent redundant calls
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True
            )
        
        # 2. Verify if the collection actually exists (helps debug deleted databases)
        if not utility.has_collection("RESUME_SEARCH"):
            st.warning("⚠️ 'RESUME_SEARCH' collection not found in Zilliz. Please check your collection name.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2 
        )

        # 3. Initialize Milvus with the active connection
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
        st.info("💡 Tip: If you are using Zilliz Cloud, ensure your URI includes the port (usually :443) and your token is current.")
        st.stop()

# Run the connection
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
