import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from prompts import RAG_PROMPT
from graph_retriever import query_graph

load_dotenv()

# Load vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)
vectorstore = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
)

def answer_question(question: str) -> str:
    graph_context = query_graph(question)

    docs = retriever.invoke(question)
    vector_context = "\n\n".join(d.page_content for d in docs)

    combined_context = f"""
    Graph Knowledge:
    {graph_context}

    Guideline Excerpts:
    {vector_context}
    """

    prompt = RAG_PROMPT.format(
        context=combined_context,
        question=question
    )

    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return response.content[0]['text']
    return response.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:\n", answer_question(q))
