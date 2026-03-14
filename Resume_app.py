import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from pymilvus import connections, utility

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's AI Career Advocate", layout="wide")

# --- 2. THE CONNECTION ENGINE (The "Test-Milvus" Integration) ---
def initialize_system():
    with st.sidebar:
        st.header("🛠 System Diagnostics")
        status_placeholder = st.empty()
        
        try:
            # STAGE 1: Physical Connection
            if not connections.has_connection("default"):
                status_placeholder.info("🔗 Connecting to Zilliz...")
                connections.connect(
                    alias="default",
                    uri=st.secrets["ZILLIZ_URI"],
                    token=st.secrets["ZILLIZ_TOKEN"],
                    secure=True,
                    timeout=30
                )
            
            # STAGE 2: Collection Verification
            if not utility.has_collection("RESUME_SEARCH"):
                st.error("❌ Collection 'RESUME_SEARCH' not found.")
                st.stop()
            
            # STAGE 3: Embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )

            # STAGE 4: LLM (Gemini 3 Flash Preview)
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                google_api_key=st.secrets["GOOGLE_API_KEY"],
                temperature=0.2
            )

            # STAGE 5: LangChain Wrapper (Using Alias-Only to prevent errors)
            v_store = Milvus(
                embedding_function=embeddings,
                connection_args={"alias": "default"}, 
                collection_name="RESUME_SEARCH"
            )
            
            status_placeholder.success("✅ System Online")
            return v_store, llm

        except Exception as e:
            status_placeholder.error(f"❌ Connection Failed: {e}")
            st.stop()

# Initialize everything
v_store, llm = initialize_system()

# --- 3. THE CLEANER ---
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

# --- 4. UI SETUP ---
st.title("🚀 Freddy Goh: Career Advocate AI")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. AGENTIC SEARCH & SYNTHESIS ---
if prompt := st.chat_input("Inquire about Freddy's technical leadership..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Phase 1: Planning
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        # Phase 2: Execution
        accumulated_context = []
        try:
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
            for topic in search_topics:
                with st.spinner(f"🔍 Searching: {topic}..."):
                    docs = retriever.invoke(topic)
                    accumulated_context.extend([d.page_content for d in docs])
        except Exception as e:
            st.warning(f"Vector search bypassed: {e}")

        # Phase 3: Final Synthesis
        context_str = "\n\n".join(list(set(accumulated_context))) if accumulated_context else "No specific resume context found."
        
        final_agent_prompt = f"""
        ROLE: You are Freddy Goh's "Career Advocate." 
        INSTRUCTION: Focus on metrics, seniority (23+ years), and leadership. 
        If a skill isn't explicitly listed, infer logically from his background.
        
        CONTEXT: {context_str}
        USER QUESTION: {prompt}
        
        TASK:
        Provide a professional, persuasive response. 
        Do not return "Career Advocate" in the response.
        Do not show any metadata or JSON.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
