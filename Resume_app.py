import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# --- IMPORTANT: UNIFIED IMPORT FROM langchain_oracledb ---
from langchain_oracledb import OracleVS, OracleHybridSearchRetriever
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
        
        # Now using the native langchain_oracledb version of OracleVS
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
        {"role": "assistant", "content": "Hello! I'm Freddy's AI assistant. How can I help you today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Hybrid Search Logic ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resume with Hybrid Search (Keywords + AI)..."):
            template = """
            SYSTEM: Expert Career Coach. Use the provided context from Freddy's resume.
            If the info is missing, mention related skills Freddy has.
            
            CONTEXT: {context}
            QUESTION: {question}
            
            INSTRUCTIONS: Provide a professional summary, list matching skills, and highlight achievements.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            try:
                # Both objects are now in the same namespace, satisfying Pydantic
                retriever = OracleHybridSearchRetriever(
                    client=conn,
                    vector_store=v_store,
                    idx_name="RES_IDX",
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
