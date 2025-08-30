# debug_retrieval_query.py
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Test the exact same setup as your API
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "admin2025"))

# Test database connection
with driver.session(database="test-db") as session:
    # Check what the retrieval system might be querying
    queries = [
        "MATCH (n:Node) RETURN count(n)",
        "MATCH (n:Text) RETURN count(n)", 
        "MATCH (n) WHERE n.name IS NOT NULL RETURN count(n)",
        "MATCH (n) WHERE n.original_text IS NOT NULL RETURN count(n)",
        "MATCH (n) RETURN keys(n) LIMIT 1"
    ]
    
    for query in queries:
        try:
            result = session.run(query)
            count = result.single()[0] if "count" in query else result.single()
            print(f"{query}: {count}")
        except Exception as e:
            print(f"{query}: ERROR - {e}")

# Test Faiss indexes
text_index = faiss.read_index("import-TEST/ATLAS-docs/vector_index/text_nodes_docs_from_json_with_emb_non_norm.index")
triple_index = faiss.read_index("import-TEST/ATLAS-docs/vector_index/triple_nodes_docs_from_json_with_emb_non_norm.index")

print(f"\nFaiss indexes:")
print(f"Text index: {text_index.ntotal} vectors")
print(f"Triple index: {triple_index.ntotal} vectors")

# Test embedding similarity
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
query = "Who is Alex Mercer?"
query_embedding = model.encode([query])

# Test Faiss search
if triple_index.ntotal > 0:
    scores, indices = triple_index.search(query_embedding.astype('float32'), k=5)
    print(f"\nFaiss search results for 'Alex Mercer':")
    print(f"Scores: {scores}")
    print(f"Indices: {indices}")

driver.close()
