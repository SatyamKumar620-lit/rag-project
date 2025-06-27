import pandas as pd
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is loaded
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Load data
df = pd.read_csv("cleaned_quotes.csv")
texts = df['quote'] + " - " + df['author']

# ChromaDB setup
# Ensure 'finetuned_model' exists locally or is a valid HuggingFace model name.
# If 'finetuned_model' is a custom fine-tuned model, make sure its path is correct
# and it's compatible with HuggingFaceEmbeddings.
embedding_model = HuggingFaceEmbeddings(model_name="finetuned_model")
vectordb = Chroma.from_texts(texts=list(texts), embedding=embedding_model, persist_directory="chroma_db")
vectordb.persist()

# Gemini model setup
# FIX: Changed model to "gemini-1.5-flash" for a faster, more cost-effective option
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# RAG chain
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 10, "lambda_mult": 0.25})
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Test query
query = "Quotes about insanity attributed to Einstein"
print(rag_chain.invoke({"query": query}))