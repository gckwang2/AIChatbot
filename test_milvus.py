import streamlit as st
from pymilvus import connections, utility, Collection
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def run_deep_diagnostic():
    with st.status("🔍 Running Deep Diagnostic...", expanded=True) as status:
        try:
            # --- PHASE 1: Connection ---
            st.write("🔗 Connecting to Zilliz Cloud...")
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True,
                timeout=20
            )
            st.success("Zilliz Connection: **OK**")

            # --- PHASE 2: Collection Check ---
            st.write("📂 Checking Collections...")
            collections = utility.list_collections()
            if "RESUME_SEARCH" not in collections:
                st.error("❌ 'RESUME_SEARCH' collection not found!")
                return
            
            # Describe the collection to check dimensions
            coll = Collection("RESUME_SEARCH")
            schema = coll.schema
            st.write(f"✅ Found 'RESUME_SEARCH' with {coll.num_entities} entities.")
            
            # --- PHASE 3: Embedding Test (The Critical Part) ---
            st.write("🤖 Testing Google Embedding Generation...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            
            test_text = "Freddy Goh Resume Diagnostic Test"
            vector = embeddings.embed_query(test_text)
            
            st.write(f"📏 Generated Vector Dimension: **{len(vector)}**")
            
            # --- PHASE 4: Schema Match Verification ---
            # Find the vector field in the schema
            vector_field = next(f for f in schema.fields if f.dtype.name == 'FLOAT_VECTOR')
            expected_dim = vector_field.params['dim']
            
            if len(vector) == expected_dim:
                st.success(f"💎 Match! Model ({len(vector)}) == Milvus ({expected_dim})")
            else:
                st.error(f"⚠️ Mismatch! Model is {len(vector)} but Milvus wants {expected_dim}")

            status.update(label="✅ Diagnostic Complete!", state="complete", expanded=False)
            st.balloons()

        except Exception as e:
            status.update(label="❌ Diagnostic Failed", state="error")
            st.error(f"Technical Traceback: {e}")

# Streamlit UI
st.title("🛡️ Milvus Health Shield")
st.info("This tool verifies the bridge between Google Embeddings and Zilliz Cloud.")

if st.button("🚀 Run Full System Test"):
    run_deep_diagnostic()
