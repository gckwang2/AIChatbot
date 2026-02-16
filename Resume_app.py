import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import PromptTemplate

# 🟢 New stable import for 2026
try:
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain.chains.retrieval_qa.base import RetrievalQA
# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's Career Advocate", layout="centered")

st.title("🚀 Freddy's AI Career Advocate")
st.caption("Advanced Reasoning RAG powered by Zilliz Cloud & Gemini 3.0 Flash Preview")

# --- 2. Connections ---
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
            temperature=0.4  # Slightly higher temperature for better "advocacy" reasoning
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
        {"role": "assistant", "content": "I am Freddy's Career Advocate. I don't just search—I connect the dots between his skills and your needs. How can I champion his profile for you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Advocate Reasoning Logic ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 🟢 THE REASONING STEP PROMPT
        template = """
        ROLE: You are Freddy Goh's "Career Advocate." Your goal is to represent Freddy's skills in the best possible light.

        REASONING STEP: 
        1. Analyze the provided resume segments below.
        2. Do not just look for keyword matches. Look for transferable skills and logical overlaps (e.g., if he knows Oracle 23ai, he understands high-performance vector databases).
        3. If a specific skill isn't listed, infer his capability based on his senior level and related technical expertise.

        CONTEXT: 
        {context}

        USER QUESTION: 
        {question}

        ADVOCATE RESPONSE: 
        Provide a persuasive summary of why Freddy is a strong fit. Highlight his technical mastery, his ability to adapt to new AI stacks, and his key achievements.
        """
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        with st.spinner("Advocating for Freddy's experience..."):
            try:
                # 🟢 Increased k=15 for a wider context window
                retriever = v_store.as_retriever(search_kwargs={"k": 15})

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = chain.invoke({"query": prompt})
                full_response = response["result"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
