import streamlit as st
import oracledb

# --- 2026 Modular Imports ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import OracleVS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document

# Use langchain-classic for the legacy RetrievalQA if you aren't ready to move to LCEL
from langchain_classic.chains import RetrievalQA

# --- 1. Page Config ---
st.set_page_config(page_title="AI Resume Architect", layout="wide")
st.title("🤖 Personal Resume Queries")
st.markdown("Tailor your 106 resume versions to any job description using Oracle AI & Gemini 1.5 Flash.")

# --- 2. Resource Caching (Prevents reconnecting on every click) ---
@st.cache_resource
def init_connections():
    # Connect to Oracle
    conn = oracledb.connect(
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        dsn=st.secrets["DB_DSN"]
    )

    # --- 1. Fix the Embeddings Model ---
    # 'text-embedding-004' is the stable standard for 2026.
    # If that fails, 'gemini-embedding-001' is the new high-performance alternative.
    embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-001", 
    google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

    # Init Embeddings
    # --- 2. Fix the Chat/Generation Model ---
    # This is where your 'gemini-3-flash-preview' belongs.
    llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    temperature=0.2
    )
    
    # 3. Init Vector Store 
    # Ensure 'embedding_function' uses the 'embeddings' variable from above
    v_store = OracleVS(
        client=conn,
        table_name="RESUME_SEARCH_V2",
        embedding_function=embeddings  # <--- This must match the variable name above
    )
    
    # 4. Init Gemini 3 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.2
    )
    
    return v_store, llm

vector_store, llm = init_connections()

# --- 3. Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Target Job Description")
    jd_text = st.text_area("Paste the JD here:", height=400, placeholder="We are looking for an Oracle Cloud expert...")
    generate_btn = st.button("Architect My Resume", type="primary")

with col2:
    st.subheader("Tailored Results")
    if generate_btn and jd_text:
        with st.spinner("Analyzing your history..."):
            # Setup RAG Chain
            template = """
            SYSTEM: Expert Career Coach. Context is from 106 resume versions.
            CONTEXT: {context}
            QUESTION: {question}
            INSTRUCTIONS: Map skills, write a summary, and extract achievements with metrics.
            """
            PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
            
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12}),
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            # Run
            result = chain.invoke(jd_text)
            st.markdown(result["result"])
    else:
        st.info("Paste a Job Description on the left and click 'Architect' to begin.")
