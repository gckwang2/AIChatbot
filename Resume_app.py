import streamlit as st
import nest_asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from pymilvus import connections, utility

# 1. FIX: Resolve async conflicts between Streamlit and Milvus
nest_asyncio.apply()

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's AI Career Assistant", layout="centered")

# --- 2. System Status UI (Always Visible) ---
st.title("🚀 Freddy's AI Career Assistant")
connection_status = st.empty()
connection_status.info("⏳ Initializing System...")

# --- 3. The Connection Engine ---
@st.cache_resource(show_spinner=False)
def get_system_resources():
    try:
        # Step A: Explicit Global Connection
        if not connections.has_connection("default"):
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True,
                timeout=60 # Increased timeout for cloud stability
            )
        
        # Step B: Load Models
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.2
        )

        # Step C: LangChain Wrapper
        v_store = Milvus(
            embedding_function=embeddings,
            connection_args={"alias": "default"},
            collection_name="RESUME_SEARCH"
        )
        
        return v_store, llm, True
    except Exception as e:
        return None, None, str(e)

# Execute Initialization
v_store, llm, status = get_system_resources()

# Update Status Message on Screen
if status is True:
    connection_status.success("✅ System Online: Connected to Zilliz Cloud")
else:
    connection_status.error(f"❌ Connection Failed: {status}")
    st.info("Check your Streamlit Secrets and ensure Zilliz cluster is active.")
    st.stop()

# --- 4. Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Freddy's Assistant. Ask me about his 23+ years of experience."}
    ]

# --- 5. Logic: The Cleaner ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        return str(response.content)
    return str(response)

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. The Research Agent ---
if prompt := st.chat_input("Ask about Freddy's expertise..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Phase 1: Search Planning
        with st.spinner("🔍 Reviewing resume data..."):
            planning_query = f"Provide 3 search terms for: {prompt}"
            plan_res = llm.invoke(planning_query)
            clean_plan = extract_clean_text(plan_res)
            topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

            # Phase 2: RAG Retrieval
            context_list = []
            retriever = v_store.as_retriever(search_kwargs={"k": 5})
            for topic in topics:
                docs = retriever.invoke(topic)
                context_list.extend([d.page_content for d in docs])
            
            context_str = "\n\n".join(list(set(context_list)))

        # Phase 3: Advocacy Response
        final_prompt = f"Using this context: {context_str}, advocate for Freddy regarding: {prompt}"
        
        with st.spinner("⚖️ Formulating response..."):
            final_res = llm.invoke(final_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
