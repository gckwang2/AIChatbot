import nest_asyncio
nest_asyncio.apply()
import streamlit as st
from pymilvus import MilvusClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's skills finder powered by AI", layout="centered")
st.title("🚀 Freddy's Skill Search powered by AI")
st.caption("Agentic RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Persistent Resources ---
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2 
    )

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

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

# --- 4. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 20+ years of experience. How can I help you today?"}
    ]

with st.status("🚀 Awakening Freddy's Career Advocate...", expanded=False) as status:
    try:
        llm = get_llm()
        embeddings_model = get_embeddings_model()
        status.update(label="✅ Systems Online", state="complete")
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

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
        # PHASE 1: Agent Research Plan
        planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{prompt}'. Output queries only, one per line."
        
        with st.spinner("🧠 Agent is planning research..."):
            plan_res = llm.invoke(planning_prompt)
            clean_plan = extract_clean_text(plan_res)
            search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

        # PHASE 2: Execution (Direct MilvusClient Search)
        accumulated_context = []
        
        try:
            st.write("🔍 Diagnostic: Initializing Direct Search...")
            # We open the client specifically for this search to avoid gRPC deadlocks
            client = MilvusClient(
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"]
            )
            st.write("✅ Diagnostic: MilvusClient Connected.")

            for topic in search_topics:
                with st.spinner(f"🔍 Searching for: {topic}..."):
                    # Generate vector manually
                    query_vector = embeddings_model.embed_query(topic)
                    
                    # Search using the lightweight client
                    results = client.search(
                        collection_name="RESUME_SEARCH",
                        data=[query_vector],
                        limit=5,
                        output_fields=["text"] 
                    )
                    
                    # Extract text chunks
                    for hit in results[0]:
                        accumulated_context.append(hit['entity']['text'])
            
            client.close() # Always close to free up the thread
            st.write("✅ Diagnostic: Search Completed & Connection Closed.")

        except Exception as e:
            st.error(f"Manual Search Error: {e}")
            st.stop()

        # PHASE 3: Synthesis & Advocacy
        context_str = "\n\n".join(list(set(accumulated_context)))
        
        final_agent_prompt = f"""
                ROLE: You are Freddy Goh's "Career Advocate." 
                CONTEXT: {context_str}
                USER QUESTION: {prompt}
                TASK: Use the context to provide a professional, persuasive response. 
                Focus on Freddy's 23+ years of experience and leadership.
                Do not show any technical metadata or JSON.
        """

        with st.spinner("⚖️ Synthesizing recommendation..."):
            final_res = llm.invoke(final_agent_prompt)
            answer = extract_clean_text(final_res)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
