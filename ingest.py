import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("vector_store")

print("vector store saved.")
