import streamlit as st
from pymilvus import connections, utility

def check_milvus_status():
    # Using 'with st.status' creates an expandable box for your logs
    with st.status("Checking Milvus Connection...", expanded=True) as status:
        try:
            st.write("🔗 Connecting to Zilliz...")
            connections.connect(
                alias="default",
                uri=st.secrets["ZILLIZ_URI"],
                token=st.secrets["ZILLIZ_TOKEN"],
                secure=True
            )
            
            st.write("✅ Connection established.")
            
            # List collections to verify access
            collections = utility.list_collections()
            st.write(f"📂 Found collections: {collections}")
            
            # Update the status header when finished
            status.update(label="Milvus Connected!", state="complete", expanded=False)
            st.toast("Milvus is ready!", icon="🚀")
            
        except Exception as e:
            status.update(label="Connection Failed", state="error")
            st.error(f"Error details: {e}")

# Call this in your app to see the visual feedback
if st.button("Run Connection Test"):
    check_milvus_status()
