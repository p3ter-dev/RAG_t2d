# Retrieval-Augmented Clinical Decision Support System for Type 2 Diabetes

This project implements an Advanced Graph-RAG (Retrieval-Augmented Generation) system designed to provide educational support for Type 2 Diabetes (T2D) management based on the **2026 ADA Standards of Care**.

## Features
- **Hybrid Retrieval:** Combines unstructured PDF data (FAISS) with structured medical knowledge (Neo4j).
- **Medical Guardrails:** Strict prompting to ensure factual, guideline-referenced responses.
- **Evaluation Suite:** Quantitative assessment of recall comparing Baseline vs. Graph RAG.

## Tech Stack
- **LLM:** Google Gemini 3 Flash
- **Embeddings:** Google Text-Embedding-004
- **Vector DB:** FAISS
- **Graph DB:** Neo4j (Running in Docker)
- **Framework:** LangChain

## Setup Instructions

1. **Prerequisites:**
   - Python 3.10+
   - Docker installed and running.
   - Google AI Studio API Key.

2. **Run Neo4j via Docker:**
   ```bash
   docker run \
       --name neo4j-rag \
       -p 7474:7474 -p 7687:7687 \
       -d \
       -e NEO4J_AUTH=neo4j/your_password \
       neo4j:latest

3. **Environment Setup: Create a .env file in the root directory:**
    GEMINI_API_KEY=your_google_api_key
    NEO4J_PASSWORD=your_password

4. **Installation:**
    pip install -r requirements.txt

5. **Data Ingestion:**
    *** 1. Populate the Graph ***
        python neo4j_ingest.py
    *** 2. Process the PDF Guidelines ***
        python ingest.py

6. **Run**
    To interact with the system in real-time:
    python rag.py

7. **Run Evaluation**
    To see the performance comparison (Baseline vs. Graph):
    python evaluation.py