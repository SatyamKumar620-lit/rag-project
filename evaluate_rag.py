from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# RAGAS (optional but shown as placeholder)
from ragas import evaluate # You'll need to install ragas if you plan to use it: pip install ragas
from ragas.metrics import faithfulness, answer_relevancy

# Load .env
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is loaded
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

# Vector DB
embedding_model = HuggingFaceEmbeddings(model_name="finetuned_model")
# Ensure 'finetuned_model' is correctly configured (local path or HF Hub name)
vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# Gemini LLM
# FIX: Changed model to "gemini-1.5-flash" to mitigate rate limit issues and follow best practices
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

# RAG Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Queries for evaluation
questions = [
    {"query": "Quotes about insanity attributed to Einstein", "ground_truth": "Einstein"},
    {"query": "Motivational quotes tagged ‘accomplishment’", "ground_truth": "accomplishment"},
    {"query": "All Oscar Wilde quotes with humor", "ground_truth": "Oscar Wilde"}
]

# Run evaluation (RAGAS placeholder - adjust if needed)
for q in questions:
    print(f"\n🔍 Query: {q['query']}")
    # FIX: Use .invoke() instead of .run() and pass the query in a dictionary
    # The output from .invoke() for RetrievalQA is a dictionary,
    # and the answer is typically under the 'result' key.
    response = qa_chain.invoke({"query": q["query"]})
    print("🧠 Gemini Response:", response["result"])

    # If you were to run RAGAS, it would typically look like this (requires more setup)
    # For RAGAS, you'd collect the responses and contexts, then create a Dataset.
    # from datasets import Dataset
    # # Example of collecting data for RAGAS
    # # You would typically store query, answer, contexts, and ground_truth
    # eval_data = {
    #     "question": [q["query"]],
    #     "answer": [response["result"]],
    #     "contexts": [response["source_documents"]], # RetrievalQA can return source_documents
    #     "ground_truth": [q["ground_truth"]] # Your ground truth for faithfulness, etc.
    # }
    # dataset = Dataset.from_dict(eval_data)
    #
    # # Then evaluate
    # # result = evaluate(
    # #     dataset,
    # #     metrics=[
    # #         faithfulness,
    # #         answer_relevancy,
    # #     ],
    # # )
    # # print(result)