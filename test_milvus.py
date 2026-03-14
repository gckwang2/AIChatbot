from pymilvus import connections, utility, MilvusException
import time

# --- CONFIGURATION ---
# Replace these with your actual values from Streamlit secrets

def test_connection():
    print(f"--- Starting Milvus Connection Test ---")
    print(f"Target URI: {ZILLIZ_URI}")
    
    start_time = time.time()
    try:
        # 1. Attempt the connection
        print("Attempting to connect...")
        connections.connect(
            alias="default",
            uri=ZILLIZ_URI,
            token=ZILLIZ_TOKEN,
            secure=True,
            timeout=30  # Increased timeout for cloud handshake
        )
        
        # 2. Verify connection status
        if connections.has_connection("default"):
            duration = round(time.time() - start_time, 2)
            print(f"✅ SUCCESS: Connected to Milvus in {duration}s")
            
            # 3. List existing collections to prove read access
            collections = utility.list_collections()
            print(f"Found {len(collections)} collections: {collections}")
            
            if "RESUME_SEARCH" in collections:
                print("✅ Found 'RESUME_SEARCH' collection.")
            else:
                print("⚠️ 'RESUME_SEARCH' not found. Check your collection name.")
                
        else:
            print("❌ FAILED: connections.has_connection returned False.")

    except MilvusException as e:
        print(f"❌ MILVUS ERROR: {e}")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
    finally:
        # Properly close
        try:
            connections.disconnect("default")
            print("--- Test Finished & Disconnected ---")
        except:
            pass

if __name__ == "__main__":
    test_connection()
