🧠 Gemini-Powered RAG-Based Semantic Quote Retrieval
This project implements a Retrieval-Augmented Generation (RAG) system for semantic quote retrieval and structured question answering using the Abirate/english_quotes dataset. It uses Google's Gemini Pro as the LLM and ChromaDB for vector storage.

🚀 Features
Fine-tuned semantic embeddings on quotes

Quote retrieval using ChromaDB with MMR (Maximal Marginal Relevance)

RAG pipeline using Gemini Pro (via LangChain)

Fully interactive Streamlit app

Structured JSON responses

Optional: Download response, see similarity context

📁 Project Structure
quote_rag_project/

├── cleaned_quotes.csv               # Preprocessed quotes

├── finetuned_model/                # Sentence-transformers model

├── chroma_db/                      # Chroma vector database

├── .env                            # Contains API key (not committed)

├── requirements.txt                # All dependencies

├── data_prep_and_finetune.py       # Fine-tunes embedding model

├── build_rag_pipeline.py           # Builds Chroma DB + RAG setup

├── evaluate_rag.py                 # Evaluates RAG with sample queries

├── app.py                          # Streamlit UI app

└── README.md                       # Project guide

⚙️ Installation
1. Clone the repo & navigate
git clone https://github.com/your-username/quote-rag
cd quote-rag

2. Install dependencies
pip install -r requirements.txt

3. Add your Gemini API key
Create a .env file:

GOOGLE_API_KEY=your_gemini_api_key_here

🔧 Usage
Step 1: Fine-tune the embedding model
python data_prep_and_finetune.py

Step 2: Build the RAG pipeline and ChromaDB
python build_rag_pipeline.py

Step 3: Evaluate the system (optional)
python evaluate_rag.py

Step 4: Launch Streamlit app
streamlit run app.py

🧪 Example Queries
"Quotes about insanity attributed to Einstein"

"Motivational quotes tagged accomplishment"

"All Oscar Wilde quotes with humor"

"Quotes tagged with both life and love by 20th-century authors"

📦 Requirements
Install all at once:

pip install -r requirements.txt

If missing, include:

langchain

langchain_community

langchain-google-genai

google-generativeai

sentence-transformers

chromadb

pandas

streamlit

python-dotenv

💡 Future Enhancements
Tag + author-year filtering (multi-hop)

Sidebar filters (e.g., by tag or author)

Visualizations of quote/tag distributions

Full RAGAS integration for auto evaluation

PLEASE REMEMBER TO ENTER YOUR API KEY IN .env file

📜 License
MIT License — feel free to fork and extend.

👤 Author
Made with ❤️ by SATYAM KUMAR PATHAK  — AI + NLP Enthusiast
