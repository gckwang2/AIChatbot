import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
# --- NEW IMPORT ---
from langchain_community.retrievers import OracleHybridSearchRetriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# ... (Keep your Page Config and CSS the same) ...

# --- 2. Connections with Auto-Reconnect ---
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
        
        # We still need the v_store for the retriever to reference
        v_store = OracleVS(
            client=conn,
            table_name="RESUME_SEARCH",
            embedding_function=embeddings
        )
        return v_store, llm, conn # Return conn for the hybrid retriever
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.stop()

vector_store, llm, conn = init_connections()

# ... (Keep Session State and History display the same) ...

# --- 4. Chat Input ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching Freddy's resume using Hybrid Search..."):
            template = """
            SYSTEM: Expert Career Coach. Use the provided context from Freddy's resume.
            If the info is missing, mention related skills Freddy has.
            
            CONTEXT: {context}
            QUESTION: {question}
            
            INSTRUCTIONS: Provide a professional summary, list matching skills, and highlight achievements.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            # --- UPDATED RETRIEVER LOGIC ---
            # Hybrid search finds both exact keywords AND similar meanings
            retriever = OracleHybridSearchRetriever(
                client=conn,
                table_name="RESUME_SEARCH",
                embeddings=embeddings,
                search_mode="hybrid", # Options: "hybrid", "keyword", or "semantic"
                k=5
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            try:
                response = chain.invoke(prompt)
                full_response = response["result"]
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
