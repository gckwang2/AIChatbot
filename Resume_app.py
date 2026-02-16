import warnings
# Suppress the Python 3.14 / Pydantic V1 compatibility warning
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality isn't compatible")

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.title("🤖 Freddy's AI Career Assistant")
st.caption("Semantic RAG powered by Zilliz Cloud & Gemini 3.0 Flash Preview")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        # Embeddings Model (Used for Semantic Vectorization)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # 🟢 Updated to use gemini-3-flash-preview
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Zilliz/Milvus Vector Store
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
        {"role": "assistant", "content": "Hello! I'm Freddy's AI career assistant, powered by Gemini 3.0. How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Semantic Retrieval Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Custom Prompt Template for Freddy's Resume
        template = """
        SYSTEM: You are a professional recruiter. Use the following pieces of context 
        from Freddy's resume to answer the question. 
        
        CONTEXT: {context}
        QUESTION: {question}
        
        INSTRUCTIONS: 
        1. Answer based strictly on the provided context.
        2. If the information isn't there, state that but highlight Freddy's core AI/Oracle strengths.
        3. Keep the tone professional and helpful.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Analyzing Freddy's resume data in Zilliz..."):
            try:
                # Semantic search: finds the top 5 chunks with the closest meaning
                retriever = v_store.as_retriever(search_kwargs={"k": 5})

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                # Running the RAG pipeline
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
