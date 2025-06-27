import streamlit as st
import os
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Gemini-Powered Quote Finder")
st.title("🧠 Gemini-Powered Quote Finder")
st.markdown("---")

# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found in environment variables. Please set it in your `.env` file.")
    st.stop()

# --- Data Loading and Processing (Cached for Performance) ---
@st.cache_data
def load_and_process_data():
    """Loads quotes from cleaned_quotes.csv and formats them."""
    try:
        df = pd.read_csv("cleaned_quotes.csv")
        # Combine quote and author for better retrieval context
        texts = df['quote'] + " - " + df['author']
        return list(texts)
    except FileNotFoundError:
        st.error("❗ Error: 'cleaned_quotes.csv' not found. Please ensure it's in the same directory as `app.py`.")
        st.stop()
    except Exception as e:
        st.error(f"❗ Error loading or processing data: {e}")
        st.stop()

texts_to_embed = load_and_process_data()

# --- Vector Database Setup (Cached for Performance) ---
@st.cache_resource
def setup_vectordb(texts_list):
    """Initializes and returns the Chroma vector database."""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="finetuned_model")
        
        vectordb = Chroma.from_texts(
            texts=texts_list,
            embedding=embedding_model,
            persist_directory="chroma_db"
        )
        return vectordb
    except Exception as e:
        st.error(f"❗ Error setting up vector database: {e}. Ensure 'finetuned_model' is correctly configured and accessible.")
        st.stop()

vectordb = setup_vectordb(texts_to_embed)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# --- Gemini LLM Setup (Cached for Performance) ---
@st.cache_resource
def setup_llm():
    """Initializes and returns the Gemini LLM."""
    try:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    except Exception as e:
        st.error(f"❗ Error initializing Gemini LLM: {e}. Please check your GOOGLE_API_KEY and model access permissions.")
        st.stop()

llm = setup_llm()

# --- RAG Chain Setup ---
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Streamlit User Interface for Querying ---
query = st.text_input(
    "📝 Enter your question about quotes (e.g., 'Quotes about courage from ancient philosophers'):",
    placeholder="Type your query here..."
)

result_data = None # To store the full response for JSON download

if query:
    with st.spinner("⏳ Searching for relevant quotes and generating response..."):
        try:
            response = qa_chain.invoke({"query": query})
            
            # Extract the generated answer
            generated_response_text = response["result"]
            
            st.markdown(f"**🧠 Gemini Response:**")
            st.write(generated_response_text)
            
            # Store full response for JSON download
            result_data = {
                "query": query,
                "response": generated_response_text,
                "retrieved_documents": []
            }

            st.markdown("---") # Separator for visual clarity

            # Optional: Show similarity scores and retrieved documents in an expander
            docs_for_display = retriever.get_relevant_documents(query)

            with st.expander("🔍 Retrieved Quotes (Context for Gemini's Answer)"):
                if not docs_for_display:
                    st.info("No highly relevant documents were found in your database for this query.")
                else:
                    displayed_quotes_content = set() # To track unique content already displayed
                    unique_docs_info = [] # To store unique documents for JSON and display

                    # Collect unique documents
                    for doc in docs_for_display:
                        if doc.page_content not in displayed_quotes_content:
                            unique_docs_info.append(doc)
                            displayed_quotes_content.add(doc.page_content)
                    
                    if not unique_docs_info: # This might happen if docs_for_display was not empty but all content was somehow weirdly duplicated or empty
                        st.info("No unique relevant documents found to display.")
                    else:
                        for i, doc in enumerate(unique_docs_info):
                            st.markdown(f"**{i+1}.** {doc.page_content}")
                            # Check if 'score' is in metadata and display if present
                            if hasattr(doc, 'metadata') and 'score' in doc.metadata:
                                st.caption(f"Similarity Score: {doc.metadata['score']:.2f}")
                            else:
                                st.caption("Similarity Score: Not available directly from this retriever configuration.")
                            result_data["retrieved_documents"].append({
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            })

            # Optional: Download JSON button
            if result_data:
                st.download_button(
                    label="📥 Download Response as JSON",
                    data=json.dumps(result_data, indent=2), # Convert dict to JSON string
                    file_name="rag_response.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.info("Please try refining your query or ensure all services are correctly configured.")