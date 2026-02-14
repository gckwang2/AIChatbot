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
st.caption("Powered by Oracle 23ai Vector Search & Gemini Flash")

# --- 2. Connections with Auto-Reconnect ---
@st.cache_resource
def get_db_connection():
    """Establishes the physical connection to Oracle."""
    return oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

def init_connections():
    """Returns the vector store and LLM, with a health check for the DB."""
    try:
        conn = get_db_connection()
        
        # --- THE FIX: Connection Health Check ---
        try:
            conn.ping() # Check if the database is still listening
        except oracledb.Error:
            # If dead, clear cache and force a fresh connection
            st.cache_resource.clear()
            conn = get_db_connection()

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Use a stable Gemini model name
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH",
            embedding_function=embeddings
        )
        return v_store, llm
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

# Initialize resources
vector_store, llm = init_connections()

# --- 3. Chat Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Freddy's AI assistant. Ask me about his technical skills or experience."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Freddy's resume..."):
            # Refined prompt for more professional output
            template = """
            SYSTEM: You are an expert Career Coach. Use the provided context from Freddy's resume to answer the question. 
            If the information isn't in the context, say you don't have that specific detail but mention related skills Freddy has.
            
            CONTEXT: {context}
            QUESTION: {question}
            
            INSTRUCTIONS: Provide a professional summary, list specific matching skills, and highlight measurable achievements.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # Re-verify connection before query to avoid DPY-4011
            init_connections() 
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 5} # k=5 is more stable for single resumes
                ),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            try:
                response = chain.invoke(prompt)
                full_response = response["result"]
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred during search: {e}")
                st.info("Try refreshing the page to reset the connection.")
