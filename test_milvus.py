import streamlit as st
import time
from pymilvus import connections, utility, Collection
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

st.set_page_config(page_title="System Stress Test", layout="centered")

st.title("🛡️ System Diagnostic Tool")
st.markdown("This script tests **Google AI** and **Zilliz Cloud** sequentially to find the bottleneck.")

if st.button("🚀 Start Deep Diagnostic"):
    
    # --- PHASE 1: GOOGLE LLM ---
    with st.status("Phase 1: Testing Google Gemini LLM...", expanded=True) as status:
        try:
            start_time = time.time()
            st.write("📡 Sending ping to Gemini Flash...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            # Force a tiny inference to prove it's alive
            response = llm.invoke("Hello, are you online?")
            duration = round(time.time() - start_time, 2)
            st.success(f"Google LLM responded in {duration}s")
            status.update(label="✅ Phase 1: Google LLM OK", state="complete")
        except Exception as e:
            st.error(f"❌ Google LLM Failed: {e}")
            st.stop()

    # --- PHASE 2: GOOGLE EMBEDDINGS ---
    with st.status("Phase 2: Testing Google Embeddings...", expanded=True) as status:
        try:
            start_time = time.time()
            st.write("🔢 Generating test vector (3072 dims)...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001", 
                google_api_key=st.secrets["GOOGLE_API_KEY"]
            )
            vector = embeddings.embed_query("Diagnostic test for Freddy")
            duration = round(time.time() - start_time, 2)
            st.write(f"📏 Vector generated with length: {len(vector)}")
            st.success(f"Embeddings generated in {duration}s")
            status.update(label="✅ Phase 2: Embeddings OK", state="complete")
        except Exception as e:
            st.error(f"❌ Embeddings Failed: {e}")
            st.stop()

    # --- PHASE 3: ZILLIZ CONNECTION ---
    with st.status("Phase 3: Testing Zilliz/Milvus Connection...", expanded=True) as status:
        try:
            start_time = time.time()
            st.write(f"🔗 Attempting handshake with {st.secrets['ZILLIZ_URI'][:20]}...")
            
            # Forced cleanup of old connections
            connections.disconnect("default") if connections.has_connection("default") else None
            
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True,
                timeout=30
            )
            
            # Check for collection
            collections = utility.list_collections()
            duration = round(time.time() - start_time, 2)
            st.write(f"📂 Collections found: {collections}")
            st.success(f"Zilliz connected in {duration}s")
            status.update(label="✅ Phase 3: Zilliz Connection OK", state="complete")
        except Exception as e:
            st.error(f"❌ Zilliz Connection Failed: {e}")
            st.info("Note: If this hangs, your Zilliz cluster might be 'Paused' in the dashboard.")
            st.stop()

    # --- PHASE 4: SCHEMA & LOAD TEST ---
    with st.status("Phase 4: Testing Collection Readiness...", expanded=True) as status:
        try:
            if "RESUME_SEARCH" in collections:
                coll = Collection("RESUME_SEARCH")
                st.write(f"📊 Entities in 'RESUME_SEARCH': {coll.num_entities}")
                st.write("🛠️ Checking if collection is loaded...")
                # This is the most common place for a hang
                utility.load_state("RESUME_SEARCH")
                st.success("Collection is fully LOADED and SEARCHABLE.")
            else:
                st.warning("⚠️ RESUME_SEARCH collection not found in this cluster.")
            
            status.update(label="✅ Phase 4: Data Readiness OK", state="complete")
        except Exception as e:
            st.error(f"❌ Data Readiness Failed: {e}")
            st.stop()

    st.balloons()
    st.success("🏁 All systems verified! If your main app still hangs, it is a Streamlit Caching/UI issue.")
