import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's AI ChatBot", layout="centered")

st.title("🚀 Freddy's AI Skill Search")
st.caption("Custom Reasoning RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's assistance. I'm ready to highlight his expertise for you!"}
    ]

# --- 3. Connections ---
@st.cache_resource
def init_connections():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.4
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

# --- 4. Helper for Deep Content Extraction ---
def get_clean_text(response):
    """Recursively extracts text from Gemini's complex response objects."""
    # If it's the standard LangChain Message
    if hasattr(response, 'content'):
        response = response.content
    
    # If it's a list (like the one that caused your error)
    if isinstance(response, list):
        # Look for the 'text' key inside the first dictionary
        if len(response) > 0 and isinstance(response[0], dict):
            return str(response[0].get('text', str(response[0])))
        return " ".join([get_clean_text(i) for i in response])
    
    # If it's a dictionary
    if isinstance(response, dict):
        return str(response.get('text', str(response)))
        
    return str(response)

# --- 5. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. Chat Input & Advocate Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # STEP A: Query Expansion
        expansion_prompt = f"Provide 3 search variations for: {prompt}. Output only the queries, one per line."
        
        with st.spinner("Widening search scope..."):
            exp_res = llm.invoke(expansion_prompt)
            # Use the deep cleaner to ensure we have a string
            exp_text = get_clean_text(exp_res)
            
            # Safe to split now
            queries = [q.strip() for q in exp_text.strip().split("\n") if q.strip()][:3]
            all_queries = [prompt] + queries

        # STEP B: Multi-Query Retrieval
        with st.spinner("Consulting Freddy's Resume via Zilliz..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                all_docs = []
                for q in all_queries:
                    all_docs.extend(retriever.invoke(q))
                
                context_text = "\n\n".join(list(set([doc.page_content for doc in all_docs])))

                # STEP C: Career Advocate Reasoning
                advocate_prompt = f"""
                ROLE: You are Freddy Goh's "Career Advocate." 
                INSTRUCTION: Analyze the resume context. Look for transferable skills and logical overlaps. 
                If a skill isn't explicitly listed, infer the related skills found in the resume,and use his previous experience based on his senior level. Do not return "Career Advocate" in the return response.
                CONTEXT: {context_text}
                QUESTION: {prompt}
                INSTRUCTION: Identify logical strengths and transferable skills. Be persuasive and professional.
                """
                
                final_res = llm.invoke(advocate_prompt)
                answer = get_clean_text(final_res)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
