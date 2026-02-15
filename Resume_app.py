import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Correct unified imports
from langchain_oracledb import OracleVS, OracleHybridSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# --- 2. Connections ---
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
        conn.ping()
        
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
            table_name="RESUME_SEARCH",
            embedding_function=embeddings
        )
        return v_store, llm, conn, embeddings
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

v_store, llm, conn, embeddings = init_connections()

# --- 3. Chat Logic ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Hybrid Search is now live. Ask me anything about Freddy's experience."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching keywords and semantic context..."):
            template = """
            SYSTEM: Expert Career Coach. Use the context from Freddy's resume.
            CONTEXT: {context}
            QUESTION: {question}
            INSTRUCTIONS: Summarize skills and achievements professionally.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            try:
                # The 'RES_IDX' must be exactly as it appears in the DB (UPPERCASE)
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
                st.markdown(response["result"])
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                
            except Exception as e:
                st.error(f"Search Error: {e}")
