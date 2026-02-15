import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
from langchain_oracledb import OracleHybridSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config & Styling ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

st.markdown("""
    <style>
    .stApp { max-width: 800px; margin: 0 auto; }
    .stChatMessage { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Freddy Goh's AI Skills Tool")
st.caption("Powered by Oracle 23ai Hybrid Search & Gemini 3 Flash")

# --- 2. Connections with Health Check ---
@st.cache_resource
def get_db_connection():
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_connections():
    try:
        conn = get_db_connection()
        # Connection Heartbeat
        try:
            conn.ping()
        except oracledb.Error:
            st.cache_resource.clear()
            conn = get_db_connection()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # v_store remains for potential metadata filtering or direct vector ops
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH",
            embedding_function=embeddings
        )
        return v_store, llm, conn, embeddings
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm, conn, embeddings = init_connections()

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Freddy's AI assistant. I now use Hybrid Search to find exact skills and experience. How can I help?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Hybrid Search Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

   # --- UPDATED RETRIEVER LOGIC (Fixed for Pydantic) ---
with st.chat_message("assistant"):
    with st.spinner("Analyzing resume with Hybrid Search..."):
        # ... (Keep prompt template as is) ...
        
        try:
            # Re-establishing the raw connection object to satisfy Pydantic
            raw_conn = get_db_connection() 
            
            # Initialize retriever
            retriever = OracleHybridSearchRetriever(
                client=raw_conn, # Use the raw connection directly
                table_name="RESUME_SEARCH",
                embeddings=embeddings,
                search_mode="hybrid", 
                k=5
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            response = chain.invoke(prompt)
            full_response = response["result"]
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Search Error: {e}")
