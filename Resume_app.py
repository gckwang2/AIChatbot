import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's AI search", layout="centered")

st.title("🚀 Freddy's AI Skill search")
st.caption("Custom Reasoning RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Using gemini-3-flash-preview
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

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's AI skill search. I'm ready to highlight his expertise for you!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Custom RAG Logic (Avoids langchain.chains) ---
# ... (Keep your previous imports and init_connections) ...

if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- NEW: QUERY EXPANSION STEP ---
        expansion_prompt = f"""
        You are an AI search optimizer. Generate 3 different versions of the following 
        user question to improve semantic search retrieval in a resume database. 
        Focus on technical synonyms and related job titles.
        
        Original Question: {prompt}
        
        Output only the 3 queries, one per line, no numbering.
        """
        
        with st.spinner("Expanding search intent..."):
            # Generate 3 variations using Gemini
            expansion_response = llm.invoke(expansion_prompt)
            # Split into a list of queries
            queries = expansion_response.content.strip().split("\n")
            # Add the original prompt to the list
            all_queries = [prompt] + queries[:3] 

        # --- RETRIEVAL STEP ---
        with st.spinner(f"Searching Milvus with {len(all_queries)} perspectives..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                all_docs = []
                
                # Search Milvus for each variation
                for q in all_queries:
                    docs = retriever.invoke(q)
                    all_docs.extend(docs)
                
                # Deduplicate documents based on content
                unique_docs = {doc.page_content: doc for doc in all_docs}.values()
                context_text = "\n\n".join([doc.page_content for doc in unique_docs])

                # --- GENERATION STEP (Career Advocate) ---
                advocate_template = """
                ROLE: Freddy Goh's Career Advocate.
                CONTEXT: {context}
                QUESTION: {question}
                INSTRUCTION: Use the context to build a persuasive case for Freddy.
                """
                
                final_prompt = advocate_template.format(context=context_text, question=prompt)
                response = llm.invoke(final_prompt)
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
                
            except Exception as e:
                st.error(f"Expansion Error: {e}")
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
