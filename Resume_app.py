import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's Career Advocate", layout="centered")

st.title("🚀 Freddy's AI Career Advocate")
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
        {"role": "assistant", "content": "I am Freddy's Career Advocate. I'm ready to highlight his expertise for you!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Custom RAG Logic (Avoids langchain.chains) ---
if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # The Career Advocate Prompt
        template = """
        ROLE: You are Freddy Goh's "Career Advocate." 
        INSTRUCTION: Analyze the resume context. Look for transferable skills and logical overlaps. 
        If a skill isn't explicitly listed, infer capability based on his senior level.
        
        CONTEXT: {context}
        
        QUESTION: {question}
        
        ADVOCATE RESPONSE:
        """
        prompt_template = ChatPromptTemplate.from_template(template)

        with st.spinner("Advocating for Freddy..."):
            try:
                # 1. Get relevant documents (k=15)
                retriever = v_store.as_retriever(search_kwargs={"k": 15})
                docs = retriever.invoke(prompt)
                
                # 2. Prepare context string
                context_text = "\n\n".join([doc.page_content for doc in docs])
                
                # 3. Create a manual chain using LCEL (No imports from langchain.chains needed!)
                chain = (
                    {"context": lambda x: context_text, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                # 4. Generate response
                full_response = chain.invoke(prompt)
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
