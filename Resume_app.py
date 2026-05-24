import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import requests
import os
import mlflow
import datetime
from mlflow.genai import evaluate
from mlflow.genai.scorers import Safety, RelevanceToQuery, Guidelines
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ==========================================
# MLFLOW & DATABRICKS CONFIGURATION
# ==========================================
os.environ["DATABRICKS_HOST"] = "https://dbc-e8b3630a-3497.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id="23305632191551")
mlflow.langchain.autolog()

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Freddy's Skills Finder", layout="centered")
st.title("🚀 Freddy's Skill Search powered by AI")
st.caption("Agentic RAG | Zilliz Cloud Vector Search | Gemini 3.0 Flash Preview")


# ==========================================
# CORE RESOURCES & LOGIC
# ==========================================
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0  
    )

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

def extract_clean_text(response):
    if hasattr(response, 'content'): content = response.content
    else: content = response
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    return str(content)

def run_agentic_rag(query: str) -> str:
    """Core logic used by UI and MLflow Evaluation."""
    llm = get_llm()
    embeddings_model = get_embeddings_model()
    
    # 1. Planning
    planning_prompt = f"Identify 3 distinct technical search queries for: '{query}'. Output only, one per line."
    plan_res = llm.invoke(planning_prompt)
    search_topics = [t.strip() for t in extract_clean_text(plan_res).split("\n") if t.strip()][:3]

    # 2. Retrieval
    accumulated_context = []
    base_uri = st.secrets["ZILLIZ_URI"].replace("https://", "").replace(":443", "")
    headers = {"Authorization": f"Bearer {st.secrets['ZILLIZ_TOKEN']}", "Content-Type": "application/json"}
    for topic in search_topics:
        payload = {"collectionName": "RESUME_SEARCH", "vector": embeddings_model.embed_query(topic), "limit": 10, "outputFields": ["text"]}
        response = requests.post(f"https://{base_uri}/v1/vector/search", headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            accumulated_context.extend([hit.get("text", "") for hit in response.json().get("data", [])])

    # 3. Synthesis
    final_agent_prompt = f"ROLE: Career Advocate. CONTEXT: {'\n\n'.join(list(set(accumulated_context)))}. QUESTION: {query}."
    return extract_clean_text(llm.invoke(final_agent_prompt))

# ==========================================
# SIDEBAR EVALUATION
# ==========================================
with st.sidebar:
    st.header("📊 Admin: MLflow Evaluation")
    if st.button("Run Evaluation"):
        with st.spinner("Running MLflow Evaluation..."):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                with mlflow.start_run(run_name=f"eval_{timestamp}"):
                    evaluate(
                        data=[{"inputs": {"query": "What is Freddy's experience with AWS?"}}],
                        predict_fn=run_agentic_rag,
                        scorers=[
                            Safety(model="endpoints:/databricks-meta-llama-3-3-70b-instruct"),
                            RelevanceToQuery(model="endpoints:/databricks-meta-llama-3-3-70b-instruct"),
                            Guidelines(model="endpoints:/databricks-meta-llama-3-3-70b-instruct", name="conciseness", guidelines="Responses must be concise.")
                        ]
                    )
                st.success("Evaluation Complete!")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# ==========================================
# MAIN CHAT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assistant ready."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        with mlflow.start_span(name="Career_Advocate_Workflow") as span:
            span.set_inputs({"user_prompt": prompt})
            
            with st.spinner("🧠 Planning search..."):
                # Logic implicitly handled by run_agentic_rag
                pass
            
            with st.spinner("⚖️ Synthesizing recommendation..."):
                answer = run_agentic_rag(prompt)
                
            span.set_outputs({"generated_answer": answer})
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
