import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's Career Advocate", layout="centered")

st.title("🚀 Freddy's AI Career Advocate")
st.caption("Advanced Reasoning RAG | Zilliz Cloud | Gemini 3.0 Flash Preview")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # 🟢 Using Gemini 3.0 Flash Preview
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.4 # Higher temperature for better reasoning/advocacy
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
        {"role": "assistant", "content": "I am Freddy's Career Advocate. I connect the dots between his deep expertise and your requirements. How can I assist?"}
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
        # 🟢 MODERN REASONING PROMPT (Career Advocate)
        system_prompt = (
            "You are Freddy Goh's 'Career Advocate.' Analyze the resume segments below. "
            "Do not just look for keyword matches; identify transferable skills and logical overlaps. "
            "If a skill isn't listed, infer capability based on his senior level and related expertise. "
            "Provide a persuasive summary of why Freddy is a strong fit.\n\n"
            "{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        with st.spinner("Advocating for Freddy's experience..."):
            try:
                # 🟢 K=15 for wide context window
                retriever = v_store.as_retriever(search_kwargs={"k": 15})

                # Create the modern retrieval chain
                question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                
                # Execute the chain
                response = rag_chain.invoke({"input": prompt})
                full_response = response["answer"]
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Reasoning Error: {e}")
