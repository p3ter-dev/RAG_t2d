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

def ingest(tx):
    tx.run("""
    MERGE (d:Disease {name: 'Type 2 Diabetes'})

    MERGE (rf1:RiskFactor {name: 'Obesity'})
    MERGE (rf2:RiskFactor {name: 'Physical Inactivity'})
    MERGE (rf3:RiskFactor {name: 'Family History'})

    MERGE (dc1:DiagnosticCriterion {name: 'HbA1c ≥ 6.5%'})
    MERGE (dc2:DiagnosticCriterion {name: 'Fasting Plasma Glucose ≥ 126 mg/dL'})

    MERGE (mg1:Management {name: 'Lifestyle Modification'})
    MERGE (mg2:Management {name: 'Regular Glycemic Monitoring'})

    MERGE (d)-[:HAS_RISK_FACTOR]->(rf1)
    MERGE (d)-[:HAS_RISK_FACTOR]->(rf2)
    MERGE (d)-[:HAS_RISK_FACTOR]->(rf3)

    MERGE (d)-[:DIAGNOSED_BY]->(dc1)
    MERGE (d)-[:DIAGNOSED_BY]->(dc2)

    MERGE (d)-[:MANAGED_BY]->(mg1)
    MERGE (d)-[:MANAGED_BY]->(mg2)
    """)

with driver.session() as session:
    session.execute_write(ingest)

driver.close()
print("Neo4j graph populated.")
