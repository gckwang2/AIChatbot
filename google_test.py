import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- 1. Page Setup ---
st.set_page_config(page_title="Google Connection Tester")
st.title("🧪 Google API Connection Test")

# --- 2. Credentials Check ---
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY not found in secrets!")
    st.stop()

# --- 3. The Cleaner (Crucial for Gemini 3) ---
def clean_output(response):
    """Handles the list-of-dicts format returned by Gemini 3 models."""
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = response
    
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict):
            return content[0].get('text', str(content[0]))
        return " ".join([str(i) for i in content])
    return str(content)

# --- 4. Testing Logic ---
st.info("🔄 Initializing Google Services...")

try:
    # TEST 1: LLM (The Brain)
    st.subheader("1. Testing LLM (gemini-3-flash-preview)")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.1
    )
    
    with st.spinner("Invoking LLM..."):
        llm_response = llm.invoke("Say 'Connection Successful' if you can read this.")
        st.success(f"Response: {clean_output(llm_response)}")

    # TEST 2: Embeddings (The Vector Engine)
    st.subheader("2. Testing Embeddings (models/embedding-001)")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    with st.spinner("Generating sample embedding..."):
        vector = embeddings.embed_query("Freddy Goh Career History")
        st.success(f"Success! Vector generated with {len(vector)} dimensions.")
        st.write(f"First 5 values: {vector[:5]}")

    st.balloons()
    st.markdown("---")
    st.write("✅ **Conclusion:** Your Google API and Key are working perfectly. The issue in the main app is likely the Milvus/Zilliz connection or the way Streamlit is handling the async loop.")

except Exception as e:
    st.error(f"❌ Test Failed: {e}")
    st.info("Check if your API Key has 'Generative Language API' enabled in the Google Cloud Console.")
