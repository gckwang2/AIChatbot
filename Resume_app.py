import streamlit as st
import oracledb
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_oracledb import OracleVS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# --- 1. Page Config & Branding ---
st.set_page_config(page_title="Freddy Goh's Executive AI", layout="centered")
st.title("💼 Freddy's Executive AI Assistant")
st.caption("Powered by Oracle 23ai & Gemini 3 Pro Reasoning")

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
        
        # 🟢 2026 Recommended Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004", # Latest stable embedding model
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # 🟢 Gemini 3 Pro for High-Quality Resume Reasoning
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", 
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            temperature=0.75, # Balanced for professional flair
            max_output_tokens=1024
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

v_store, llm = init_connections()

# --- 3. Professional System Prompt ---
# This turns the model from a basic chatbot into an Executive Recruiter
template = """
SYSTEM: You are an Elite Executive Recruiter. Your task is to analyze Freddy's resume 
and answer questions with high-impact, professional language. 

STRATEGY:
- Use strong action verbs (Spearheaded, Optimized, Orchestrated).
- Highlight quantifiable achievements from the context.
- If the information isn't present, highlight a relevant transferable skill.

CONTEXT: {context}
QUESTION: {question}

PROFESSIONAL SUMMARY:
"""
prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome. I am Freddy's AI Career Orchestrator. How can I demonstrate his value to your organization today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Freddy's expertise..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing resume archives..."):
            try:
                # 🟢 VECTOR SEARCH (Works on Oracle Free Tier)
                retriever = v_store.as_retriever(search_kwargs={"k": 4})

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": prompt_template}
                )
                
                response = qa
