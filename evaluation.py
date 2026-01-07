import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from graph_retriever import query_graph

load_dotenv()

# vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    api_key=os.getenv("GEMINI_API_KEY")
)

vectorstore = FAISS.load_local(
    "vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    api_key=os.getenv("GEMINI_API_KEY")
)

# Evaluation Data
evaluation_set = [
    {
        "question": "What are the risk factors for Type 2 Diabetes?",
        "gold": ["obesity", "physical inactivity", "family history"]
    },
    {
        "question": "How is Type 2 Diabetes diagnosed?",
        "gold": ["hbA1c", "fasting plasma glucose"]
    },
    {
        "question": "How is Type 2 Diabetes managed?",
        "gold": ["lifestyle", "glycemic monitoring"]
    }
]

# Recall
def calculate_recall(retrieved_text: str, gold_terms: list[str]) -> float:
    retrieved_text = retrieved_text.lower()
    hits = sum(1 for term in gold_terms if term.lower() in retrieved_text)
    return hits / len(gold_terms)

# Baseline RAG (Vector only)
def baseline_rag(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        f"Answer the question using the context below:\n{context}\n\nQuestion: {question}"
    )

    if isinstance(response.content, list):
        return response.content[0]["text"]

    return response.content

# Graph RAG (Vector + Neo4j)
def graph_rag(question: str) -> str:
    graph_info = query_graph(question)
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    response = llm.invoke(
        f"""
Structured medical knowledge:
{graph_info}

Clinical guideline context:
{context}

Question:
{question}
"""
    )

    if isinstance(response.content, list):
        return response.content[0]["text"]

    return response.content

# evaluation
baseline_scores = []
graph_scores = []

for item in evaluation_set:
    print(f"\nEvaluating: {item['question']}")

    baseline_answer = baseline_rag(item["question"])
    graph_answer = graph_rag(item["question"])

    baseline_recall = calculate_recall(baseline_answer, item["gold"])
    graph_recall = calculate_recall(graph_answer, item["gold"])

    baseline_scores.append(baseline_recall)
    graph_scores.append(graph_recall)

    print(f"Baseline Recall: {baseline_recall:.2f}")
    print(f"Graph RAG Recall: {graph_recall:.2f}")

print("\nFINAL RESULTS")
print(f"Average Baseline Recall: {sum(baseline_scores)/len(baseline_scores):.2f}")
print(f"Average Graph RAG Recall: {sum(graph_scores)/len(graph_scores):.2f}")