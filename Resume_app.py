import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_milvus import Milvus
from langchain_core.messages import AIMessage

# ... (Keep your existing init_connections and Page Config) ...

if prompt := st.chat_input("Ask about Freddy's potential..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- 1. QUERY EXPANSION STEP ---
        expansion_prompt = f"""
        Generate 3 different search queries to find resumes related to this question.
        Focus on technical synonyms and skills.
        
        Original Question: {prompt}
        
        Output only the 3 queries, one per line, no numbering.
        """
        
        with st.spinner("Expanding search intent..."):
            expansion_response = llm.invoke(expansion_prompt)
            
            # 🟢 Robustly handle the response content
            if isinstance(expansion_response, AIMessage):
                expansion_text = expansion_response.content
            else:
                expansion_text = str(expansion_response)

            # Clean and split (This is where your error was happening)
            queries = [q.strip() for q in expansion_text.split("\n") if q.strip()]
            all_queries = [prompt] + queries[:3] 

        # --- 2. MULTI-QUERY RETRIEVAL ---
        with st.spinner(f"Searching Milvus for '{prompt}' and synonyms..."):
            try:
                retriever = v_store.as_retriever(search_kwargs={"k": 5})
                all_docs = []
                
                # Search for each variation to widen the net
                for q in all_queries:
                    docs = retriever.invoke(q)
                    all_docs.extend(docs)
                
                # Deduplicate documents
                unique_docs = {doc.page_content: doc for doc in all_docs}.values()
                context_text = "\n\n".join([doc.page_content for doc in unique_docs])

                # --- 3. CAREER ADVOCATE REASONING ---
                advocate_prompt = f"""
                ROLE: You are Freddy Goh's Career Advocate.
                CONTEXT: {context_text}
                QUESTION: {prompt}
                
                INSTRUCTION: Analyze the context. Highlight Freddy's skills and infer 
                logical strengths based on his experience. Be persuasive.
                """
                
                final_response = llm.invoke(advocate_prompt)
                
                # Extract content safely
                answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Search/Advocacy Error: {e}")
