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
# This mimics the chat memory of Gemini
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Freddy's AI assistant. Paste a job description or ask me about Freddy's technical skills."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input (The Gemini bar at the bottom) ---
if prompt := st.chat_input("Ask about Freddy's skills..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
if prompt := st.chat_input("Ask about Freddy's skills..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Create a placeholder for the "typing" effect
        response_placeholder = st.empty()
        full_response = ""
        
        # We use a simple spinner while the search happens
        with st.spinner("Searching Freddy's history..."):
            # Update k=5 and similarity for speed
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Streaming the result
            # Note: For legacy RetrievalQA, it returns the whole block, 
            # but using st.write_stream makes it feel smoother.
            result = chain.invoke(prompt)
            full_response = result["result"]
            
        # This gives that Gemini "fade-in" or typing effect
        st.write_stream(iter(full_response.split(" "))) 
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
