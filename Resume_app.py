import nest_asyncio
nest_asyncio.apply()
import streamlit as st
import requests
import os
import mlflow
from mlflow.genai import evaluate
from mlflow.genai.scorers import Safety, RelevanceToQuery, Guidelines
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- MLflow & Databricks Configuration ---
os.environ["DATABRICKS_HOST"] = "https://dbc-e8b3630a-3497.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = st.secrets["DATABRICKS_TOKEN"]
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(experiment_id="23305632191551")
mlflow.langchain.autolog()

# --- 1. Page Config ---
st.set_page_config(page_title="Freddy's skills finder powered by AI", layout="centered")
st.title("🚀 Freddy's Skill Search powered by AI")
st.caption("Agentic RAG | Zilliz Cloud Vector Search | Gemini 3.0 Flash Preview")

# --- 2. Persistent Resources ---
@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0  # Enforcing deterministic, grounded output
    )

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )

# --- 3. THE CLEANER ---
def extract_clean_text(response):
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    return str(content)

# --- 4. Core RAG Logic (Decoupled from UI) ---
def run_agentic_rag(query: str) -> str:
    """Pure logic function for both UI and MLflow Evaluation."""
    llm = get_llm()
    embeddings_model = get_embeddings_model()
    
    # PHASE 1: Plan
    planning_prompt = f"Identify 3 distinct technical search queries to evaluate: '{query}'. Output queries only, one per line."
    plan_res = llm.invoke(planning_prompt)
    clean_plan = extract_clean_text(plan_res)
    search_topics = [t.strip() for t in clean_plan.split("\n") if t.strip()][:3]

    # PHASE 2: Search
    accumulated_context = []
    base_uri = st.secrets["ZILLIZ_URI"].replace("https://", "").replace(":443", "")
    search_url = f"https://{base_uri}/v1/vector/search"
    headers = {
        "Authorization": f"Bearer {st.secrets['ZILLIZ_TOKEN']}",
        "Content-Type": "application/json"
    }

    for topic in search_topics:
        query_vector = embeddings_model.embed_query(topic)
        payload = {
            "collectionName": "RESUME_SEARCH",
            "vector": query_vector,
            "limit": 10,
            "outputFields": ["text"]
        }
        
        response = requests.post(search_url, headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            results = response.json().get("data", [])
            for hit in results:
                accumulated_context.append(hit.get("text", ""))

    # PHASE 3: Synthesis
    context_str = "\n\n".join(list(set(accumulated_context)))
    final_agent_prompt = f"""
            ROLE: You are Freddy Goh's "Career Advocate." 
            CONTEXT: {context_str}
            USER QUESTION: {query}
            TASK:
    Use the context to provide a professional, persuasive response. 
    Focus on metrics, seniority (23+ years), and leadership. 
    If a skill isn't explicitly listed, infer the related skills found in the resume, and use his previous experience based on his senior level. Do not return "Career Advocate" in the return response.
    Do not show any metadata, JSON, or technical signatures.
    """
    
    final_res = llm.invoke(final_agent_prompt)
    return extract_clean_text(final_res)

# --- 5. Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "I am Freddy's Assistant. I've analyzed his 23+ years of experience. How can I help you today?"}
    ]

with st.status("🚀 Awakening Freddy's Career Advocate...", expanded=False) as status:
    try:
        # Pre-warm resources
        get_llm()
        get_embeddings_model()
        status.update(label="✅ Systems Online", state="complete")
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

# --- 6. MLflow Evaluation Sidebar ---
with st.sidebar:
    st.header("📊 Admin: MLflow Evaluation")
    st.write("Run GenAI Judges against the RAG pipeline.")
    
    if st.button("Run Evaluation Dataset"):
        with st.spinner("Running MLflow Evaluation..."):
            # Step 1: Define evaluation dataset
            eval_dataset = [{
                "inputs": {"query": "What is Freddy's experience with AWS and Cloud Architecture?"}
            }]

            # Step 2: Define predict_fn wrapper
            def predict(query):
                return run_agentic_rag(query)

            # Step 3: Run evaluation
            try:
                results = evaluate(
                    data=eval_dataset,
                    predict_fn=predict,
                    scorers=[
                        Safety(),
                        RelevanceToQuery(),
                        Guidelines(name="conciseness", guidelines="Responses must be concise."),
                    ],
                )
                st.success("Evaluation Complete! Check Databricks UI.")
                st.write(f"Run ID: {results.run_id}")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

# --- 7. Main Chat UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with mlflow.start_span(name="Career_Advocate_Workflow") as span:
            span.set_inputs({"user_prompt": prompt})
            
            with st.spinner("⚖️ Synthesizing recommendation..."):
                # Call the decoupled RAG logic
                answer = run_agentic_rag(prompt)
                
            span.set_outputs({"generated_answer": answer})
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
