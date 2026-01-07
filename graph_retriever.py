import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(
    URI, auth=(USERNAME, PASSWORD)
)

def query_graph(question: str) -> str:
    q = question.lower()

    with driver.session() as session:
        if "risk" in q:
            result = session.run("""
                MATCH (d:Disease)-[:HAS_RISK_FACTOR]->(r)
                RETURN r.name AS info
            """)
        elif "diagnos" in q:
            result = session.run("""
                MATCH (d:Disease)-[:DIAGNOSED_BY]->(c)
                RETURN c.name AS info
            """)
        elif "manage" in q or "lifestyle" in q:
            result = session.run("""
                MATCH (d:Disease)-[:MANAGED_BY]->(m)
                RETURN m.name AS info
            """)
        else:
            return ""

        return "\n".join(record["info"] for record in result)
