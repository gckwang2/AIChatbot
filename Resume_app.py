import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's Career Advocate", layout="centered")

# --- 2. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Career Advocate. I'm ready to highlight his expertise for you!"}
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

# --- 4. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. Helper Function for Content Extraction ---
def extract_text(response):
    """Safely extracts string content from any LLM response type."""
    if hasattr(response, 'content'):
        return str(response.content)
    if isinstance(response, dict) and 'content' in response:
        return str(response['content'])
    if isinstance(response, list) and len(response) > 0:
        return extract_text(response[0])
    return str(response)

# --- 6. Chat Input & Advocate Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- STEP A: Query Expansion ---
        expansion_prompt = f"Provide 3 search variations for: {prompt}. Output only the queries, one per line."
        
        with st.spinner("Widening search..."):
            exp_res = llm.invoke(expansion_prompt)
            # Use the fail-proof extraction helper
            exp_text = extract_text(exp_res)
            
            # Now .split() is guaranteed to work on a string
            queries = [q.strip() for q in exp_text.strip().split("\n") if q.strip()][:3]
            all_queries = [prompt] + queries

        # --- STEP B: Multi-Query Retrieval ---
        with st.spinner("Analyzing Zilliz data..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                all_docs = []
                for q in all_queries:
                    # Using invoke directly on retriever for modern LCEL compatibility
                    all_docs.extend(retriever.invoke(q))
                
                # Deduplicate based on content to save tokens
                unique_contents = list(set([doc.page_content for doc in all_docs]))
                context_text = "\n\n".join(unique_contents)

                # --- STEP C: Career Advocate Reasoning ---
                advocate_prompt = f"""
                ROLE: Freddy Goh's Career Advocate.
                CONTEXT: {context_text}
                QUESTION: {prompt}
                INSTRUCTION: Identify logical strengths and transferable skills. Be persuasive.
                """
                
                final_res = llm.invoke(advocate_prompt)
                answer = extract_text(final_res)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
