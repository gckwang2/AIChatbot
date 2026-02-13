import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy Goh's AI Skills", layout="centered")

# Gemini-style CSS for clean aesthetics
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
        # UPDATED: Using text-embedding-004 for better stability with new keys
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-001", 
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
if prompt := st.chat_input("Ask about Freddy's skills...", key="freddy_ai_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = "" # Initialize to prevent NameError
        with st.spinner("Analyzing..."):
            try:
                template = """
                SYSTEM: Expert Career Coach. Context is from Freddy's resume.
                CONTEXT: {context}
                QUESTION: {question}
                INSTRUCTIONS: Map skills, write a summary, and extract achievements with metrics.
                """
                PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
                
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                response = chain.invoke(prompt)
                full_response = response["result"]
                
                # Streaming effect
                st.write_stream(iter(full_response.split(" ")))
                
                # Save to history ONLY if successful
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                error_msg = f"⚠️ API Error: {str(e)}"
                if "API key expired" in error_msg:
                    st.error("Your Google API Key has expired. Please update it in Streamlit Secrets.")
                else:
                    st.error(f"An unexpected error occurred: {e}")
