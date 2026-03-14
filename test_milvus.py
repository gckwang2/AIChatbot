from pymilvus import connections, utility, MilvusException
import time
import os

def test_connection():
    # --- CONFIGURATION ---
    # Try to get from environment/secrets first (for Cloud), 
    # otherwise fallback to hardcoded strings (for local testing)
    try:
        import streamlit as st
        uri = st.secrets.get("ZILLIZ_URI", "YOUR_HARDCODED_URI_HERE")
        token = st.secrets.get("ZILLIZ_TOKEN", "YOUR_HARDCODED_TOKEN_HERE")
    except:
        # If not running in streamlit, use these:
        uri = "https://your-endpoint.zillizcloud.com:443" 
        token = "your-api-key-token"

    print(f"--- Starting Milvus Connection Test ---")
    print(f"Target URI: {uri}")
    
    start_time = time.time()
    try:
        # 1. Attempt the connection
        print("Attempting to connect...")
        connections.connect(
            alias="default",
            uri=uri,
            token=token,
            secure=True,
            timeout=30 
        )
        
        # 2. Verify connection status
        if connections.has_connection("default"):
            duration = round(time.time() - start_time, 2)
            print(f"✅ SUCCESS: Connected to Milvus in {duration}s")
            
            # 3. List existing collections
            collections = utility.list_collections()
            print(f"Found {len(collections)} collections: {collections}")
                
        else:
            print("❌ FAILED: connections.has_connection returned False.")

    except MilvusException as e:
        print(f"❌ MILVUS ERROR: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    finally:
        try:
            connections.disconnect("default")
            print("--- Test Finished & Disconnected ---")
        except:
            pass

if __name__ == "__main__":
    test_connection()
