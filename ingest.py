import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "data/standards-of-care-2026.pdf"
)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GEMINI_API_KEY")
)

# Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vector_store")

print("vector store saved.")
