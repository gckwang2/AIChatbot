import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.markdown("""
    <style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .stChatMessage { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Freddy Goh's AI Skills Tool")
st.caption("Powered by Oracle 23ai Vector Search & Gemini 3 Flash")

# --- 2. Connections ---
@st.cache_resource
def init_connections():
    try:
        conn = oracledb.connect(
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            dsn=st.secrets["DB_DSN"]
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH_V2",
            embedding_function=embeddings
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

vector_store, llm = init_connections()

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Freddy's AI assistant. Ask me about Freddy's technical skills."}
    ]

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input ---
# Added a unique key to prevent the Duplicate ID error
if prompt := st.chat_input("Ask about Freddy's skills...", key="freddy_ai_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            template = """
            SYSTEM: Expert Career Coach. Context is from Freddy's resume.
            CONTEXT: {context}
            QUESTION: {question}
            INSTRUCTIONS: Map skills, write a summary, and extract achievements with metrics.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # FIXED: search_type="similarity" and k=5 for instant speed
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            response = chain.invoke(prompt)
            full_response = response["result"]
            
            # ADDED: Streaming effect for a better UX
            st.write_stream(iter(full_response.split(" ")))
            
    # FIXED: This line is now INSIDE the if-block to prevent the crash
    st.session_state.messages.append({"role": "assistant", "content": full_response})
